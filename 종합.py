"""
할리스 매장 밀도 분석 파이프라인 (노트북 -> 단일 py 스크립트 변환)

실행:
    python hollys_analysis.py                # 전체 단계 실행
    python hollys_analysis.py --skip-crawl    # 크롤링 생략(이미 source/hollys_store.csv 있을 때)
    python hollys_analysis.py --skip-geocode  # 좌표변환 생략(이미 결과 csv 있을 때, KAKAO_API_KEY 불필요)

필요 파일 (source/ 폴더에 미리 준비):
    - 행정구역_시군구_별__성별_인구수.csv   (KOSIS 인구 원본)
    - skorea-provinces-2018-geo.json        (시도 GeoJSON)
    - .env 파일에 KAKAO_API_KEY=xxxx         (좌표변환 시 필요)

결과:
    - output/hollys_report.csv, hollys_barplot.png, hollys_density_map.html
    - output/report.html  <- 모든 결과를 모아 보는 최종 HTML 리포트 (더블클릭으로 브라우저에서 확인)
"""

import argparse
import base64
import datetime
import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")  # 스크립트 실행 시 화면 표시 대신 파일 저장만 함
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from tqdm import tqdm

try:
    import folium
except ImportError:
    folium = None

SOURCE_DIR = "source"
OUTPUT_DIR = "output"
BASE_URL = "https://www.hollys.co.kr/store/korea/korStore2.do"


def ensure_dirs():
    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_korean_font():
    """OS별로 사용 가능한 한글 폰트를 자동 선택 (원본 노트북은 AppleGothic 고정이었음)."""
    import matplotlib.font_manager as fm
    candidates = ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


# =====================================================
# 1단계: 할리스 매장 크롤링
# =====================================================
def parse_paging_info(soup):
    paging_div = soup.select_one("div.paging")
    if paging_div is None:
        return [], None

    page_numbers = []
    for tag in paging_div.select("a, strong"):
        txt = tag.get_text(strip=True)
        if txt.isdigit():
            page_numbers.append(int(txt))

    next_block_page = None
    for a in paging_div.select("a[onclick]"):
        onclick_text = a.get("onclick")
        match = re.search(r"paging\((\d+)\s*,\s*1\)", onclick_text)
        if match:
            next_block_page = int(match.group(1))
            break

    return page_numbers, next_block_page


def get_total_pages():
    page = 1
    max_page = 1
    while True:
        print(f"총페이지 탐색중... (현재 확인 페이지: {page})")
        res = requests.get(BASE_URL, params={"pageNo": page})
        soup = BeautifulSoup(res.text, "html.parser")
        page_numbers, next_block_page = parse_paging_info(soup)
        if page_numbers:
            max_page = max(max_page, max(page_numbers))
        if next_block_page is None:
            break
        page = next_block_page
        time.sleep(0.2)
    print("최종 확인된 총 페이지 수:", max_page)
    return max_page


def crawl_store_page(page):
    res = requests.get(BASE_URL, params={"pageNo": page})
    if res.status_code != 200:
        print(f"{page}페이지 요청 실패:", res.status_code)
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    tbody = soup.select_one("table.tb_store tbody")
    if tbody is None:
        return []

    page_result = []
    for row in tbody.select("tr"):
        tds = row.select("td")
        if len(tds) < 6:
            continue

        area = tds[0].get_text(strip=True)
        name = tds[1].get_text(strip=True)
        status = tds[2].get_text(strip=True)
        addr = tds[3].get_text(strip=True)

        service_list = [img.get("alt").strip() for img in tds[4].select("img") if img.get("alt")]
        store_service = "/".join(service_list)
        phone = tds[5].get_text(strip=True)

        page_result.append([area, name, status, addr, store_service, phone])

    return page_result


def step1_crawl():
    total_pages = get_total_pages()
    all_data = []
    for page in range(1, total_pages + 1):
        print(f"매장 수집중: {page}/{total_pages}")
        all_data.extend(crawl_store_page(page))
        time.sleep(0.3)

    df = pd.DataFrame(all_data, columns=["지역", "매장명", "현황", "주소", "매장서비스", "전화번호"])
    print("\n최종 매장 수:", len(df))

    df.to_csv(f"{SOURCE_DIR}/hollys_store.csv", index=False, encoding="utf-8")
    print("저장 완료:", f"{SOURCE_DIR}/hollys_store.csv")
    return df


# =====================================================
# 2단계: 주소 -> 위도/경도 (카카오 Geocoding)
# =====================================================
def clean_address(address):
    if pd.isna(address):
        return ""
    addr = str(address)
    addr = re.sub(r"\(.*?\)", "", addr)
    addr = addr.split(",")[0]
    for pattern in [r"\d+\s*층", r"\d+\s*호", r"지하\s*\d*", r"B\d+", r"\d+F", r"\d+~\d+층", r"\d+~\d+", r"\s*층"]:
        addr = re.sub(pattern, "", addr)
    addr = addr.replace("·", " ").replace(".", " ")
    addr = re.sub(r"\s+", " ", addr)
    return addr.strip()


def kakao_address_search(query, api_key):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    response = requests.get(url, headers=headers, params={"query": query})
    if response.status_code != 200:
        print("주소검색 요청 실패:", response.status_code, response.text)
        return None, None
    result = response.json()
    if result["documents"]:
        d = result["documents"][0]
        return float(d["y"]), float(d["x"])
    return None, None


def kakao_keyword_search(query, api_key):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    response = requests.get(url, headers=headers, params={"query": query})
    if response.status_code != 200:
        print("키워드검색 요청 실패:", response.status_code, response.text)
        return None, None
    result = response.json()
    if result["documents"]:
        d = result["documents"][0]
        return float(d["y"]), float(d["x"])
    return None, None


def extract_rest_area(store_name):
    rest_name = store_name.replace("(상)", "").replace("(하)", "")
    rest_name = rest_name.replace("휴게소점", "휴게소")
    return rest_name.strip()


SIDO_MAP = {
    "서울": "서울특별시", "서울시": "서울특별시", "서울특별시": "서울특별시",
    "부산": "부산광역시", "부산시": "부산광역시", "부산광역시": "부산광역시",
    "대구": "대구광역시", "대구시": "대구광역시", "대구광역시": "대구광역시",
    "인천": "인천광역시", "인천시": "인천광역시", "인천광역시": "인천광역시",
    "광주": "광주광역시", "광주시": "광주광역시", "광주광역시": "광주광역시",
    "대전": "대전광역시", "대전시": "대전광역시", "대전광역시": "대전광역시",
    "울산": "울산광역시", "울산시": "울산광역시", "울산광역시": "울산광역시",
    "세종": "세종특별자치시", "세종시": "세종특별자치시", "세종특별자치시": "세종특별자치시",
    "경기": "경기도", "경기도": "경기도",
    "강원": "강원특별자치도", "강원도": "강원특별자치도", "강원특별자치도": "강원특별자치도",
    "충북": "충청북도", "충청북도": "충청북도",
    "충남": "충청남도", "충청남도": "충청남도",
    "전북": "전북특별자치도", "전라북도": "전북특별자치도", "전북특별자치도": "전북특별자치도",
    "전남": "전라남도", "전라남도": "전라남도",
    "경북": "경상북도", "경상북도": "경상북도",
    "경남": "경상남도", "경상남도": "경상남도",
    "제주": "제주특별자치도", "제주도": "제주특별자치도", "제주특별자치도": "제주특별자치도",
}


def step2_geocode():
    load_dotenv()
    api_key = os.getenv("KAKAO_API_KEY")
    if not api_key:
        raise RuntimeError("KAKAO_API_KEY가 .env에 없습니다. .env 파일에 KAKAO_API_KEY=값 을 추가하세요.")

    df = pd.read_csv(f"{SOURCE_DIR}/hollys_store.csv")

    lat_list, lon_list, clean_addr_list, method_list = [], [], [], []

    for store, addr in tqdm(zip(df["매장명"], df["주소"]), total=len(df)):
        cleaned_addr = clean_address(addr)
        clean_addr_list.append(cleaned_addr)

        lat, lon = kakao_address_search(cleaned_addr, api_key)
        if lat is not None:
            lat_list.append(lat); lon_list.append(lon); method_list.append("주소검색")
            time.sleep(0.2)
            continue

        keyword = (extract_rest_area(store) + " 할리스") if "휴게소" in store else (store + " 할리스")
        lat, lon = kakao_keyword_search(keyword, api_key)
        if lat is not None:
            lat_list.append(lat); lon_list.append(lon); method_list.append("키워드검색")
        else:
            lat_list.append(None); lon_list.append(None); method_list.append("실패")
        time.sleep(0.2)

    df["주소_전처리"] = clean_addr_list
    df["위도"] = lat_list
    df["경도"] = lon_list
    df["검색방식"] = method_list

    print("좌표 변환 성공률:", df["위도"].notnull().mean())

    if "시도" not in df.columns:
        df["시도"] = df["주소"].astype(str).str.split().str[0]
    df["시도"] = df["시도"].replace(SIDO_MAP)

    df.to_csv(f"{SOURCE_DIR}/hollys_store_geo_kakao_final.csv", index=False, encoding="utf-8")
    print("저장 완료:", f"{SOURCE_DIR}/hollys_store_geo_kakao_final.csv")
    return df


# =====================================================
# 3~5단계: 시도별 매장수 / 인구 병합 / 인구대비 매장수
# =====================================================
def step3_store_count():
    df_store = pd.read_csv(f"{SOURCE_DIR}/hollys_store_geo_kakao_final.csv")
    store_count = df_store["시도"].value_counts().reset_index()
    store_count.columns = ["시도", "매장수"]
    return store_count


def step4_population():
    df = pd.read_csv(f"{SOURCE_DIR}/행정구역_시군구_별__성별_인구수.csv", encoding="utf-8")
    df = df[~df["행정구역(시군구)별"].isin(["행정구역(시군구)별", "전국"])].copy()
    df = df[["행정구역(시군구)별", "2026.06"]]
    df = df.rename(columns={"행정구역(시군구)별": "시도", "2026.06": "인구"})
    df["인구"] = pd.to_numeric(df["인구"], errors="coerce")
    df = df.dropna(subset=["인구"])
    df["인구"] = df["인구"].astype(int)
    df = df.reset_index(drop=True)
    df.to_csv(f"{SOURCE_DIR}/population_sido.csv", index=False, encoding="utf-8-sig")
    print("저장 완료:", f"{SOURCE_DIR}/population_sido.csv")
    return df


def step5_merge(store_count, df_pop):
    df_merge = store_count.merge(df_pop, on="시도", how="inner")
    df_merge["10만명당_매장수"] = (df_merge["매장수"] / df_merge["인구"]) * 100000
    df_merge = df_merge.sort_values("10만명당_매장수", ascending=False)
    df_merge.to_csv(f"{SOURCE_DIR}/hollys_population_analysis.csv", index=False, encoding="utf-8-sig")
    print("저장 완료:", f"{SOURCE_DIR}/hollys_population_analysis.csv")
    return df_merge


def step_cluster():
    df = pd.read_csv(f"{SOURCE_DIR}/hollys_store_geo_kakao_final.csv")
    df = df.dropna(subset=["위도", "경도"]).reset_index(drop=True)
    coords = df[["위도", "경도"]].values

    kms_per_radian = 6371.0088
    epsilon = 0.8 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=5, algorithm="ball_tree", metric="haversine")
    df["cluster"] = db.fit_predict(np.radians(coords))

    df.to_csv(f"{SOURCE_DIR}/hollys_cluster.csv", index=False, encoding="utf-8-sig")
    print("저장 완료:", f"{SOURCE_DIR}/hollys_cluster.csv")
    return df


# =====================================================
# 6~7단계: 리포트 표 / 그래프
# =====================================================
def step6_report(df_merge):
    df_merge["인구(만명)"] = df_merge["인구"] / 10000
    df_report = df_merge[["시도", "매장수", "인구(만명)", "10만명당_매장수"]].round(2)
    df_report.to_csv(f"{OUTPUT_DIR}/hollys_report.csv", index=False, encoding="utf-8-sig")
    print("저장 완료:", f"{OUTPUT_DIR}/hollys_report.csv")
    return df_report


def step7_barplot(df_merge):
    set_korean_font()
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_merge, hue="시도", x="시도", y="10만명당_매장수", legend=False)
    plt.xticks(rotation=45)
    plt.title("시도별 인구 10만명당 할리스 매장 수")
    plt.xlabel("시도")
    plt.ylabel("10만명당 매장 수")
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2, p.get_height(), f"{p.get_height():.2f}",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/hollys_barplot.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print("저장 완료:", path)
    return path


def step7b_scatter(df_merge):
    set_korean_font()
    plt.figure(figsize=(8, 6))
    plt.scatter(df_merge["인구"], df_merge["매장수"])
    for _, row in df_merge.iterrows():
        plt.text(row["인구"], row["매장수"], row["시도"], fontsize=9)
    plt.title("시도별 인구와 할리스 매장 수 관계")
    plt.xlabel("인구")
    plt.ylabel("매장 수")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/hollys_scatter.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print("저장 완료:", path)
    return path


# =====================================================
# 8단계: Choropleth 지도 (HTML)
# =====================================================
def step8_map(df_merge):
    if folium is None:
        print("folium 미설치로 지도 생성을 건너뜁니다. `pip install folium` 후 재실행하세요.")
        return None

    geo_path = f"{SOURCE_DIR}/skorea-provinces-2018-geo.json"
    with open(geo_path, encoding="utf-8") as f:
        geo = json.load(f)

    name_fix = {"강원도": "강원특별자치도", "전라북도": "전북특별자치도"}
    for feat in geo["features"]:
        n = feat["properties"]["name"]
        feat["properties"]["name"] = name_fix.get(n, n)

    m = folium.Map(location=[35.9, 127.8], zoom_start=7)
    folium.Choropleth(
        geo_data=geo,
        data=df_merge,
        columns=["시도", "10만명당_매장수"],
        key_on="feature.properties.name",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name="10만명당 할리스 매장 수",
    ).add_to(m)

    path = f"{OUTPUT_DIR}/hollys_density_map.html"
    m.save(path)
    print("저장 완료:", path)
    return path


# =====================================================
# 최종 HTML 리포트 (모든 결과를 하나의 파일로 통합)
# =====================================================
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_html_report(df_report, bar_png, scatter_png, map_html):
    now = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")

    table_html = df_report.to_html(index=False, classes="tbl", border=0)
    bar_b64 = img_to_base64(bar_png) if bar_png and os.path.exists(bar_png) else None
    scatter_b64 = img_to_base64(scatter_png) if scatter_png and os.path.exists(scatter_png) else None

    map_iframe = ""
    if map_html and os.path.exists(map_html):
        map_iframe = f'<iframe src="{os.path.basename(map_html)}" width="100%" height="600" style="border:1px solid #ddd;"></iframe>'

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>할리스 매장 밀도 분석 리포트</title>
<style>
  body {{ font-family: "Malgun Gothic","AppleGothic",sans-serif; max-width: 900px; margin: 40px auto; color: #222; }}
  h1 {{ font-size: 22px; border-bottom: 2px solid #333; padding-bottom: 8px; }}
  h2 {{ font-size: 18px; margin-top: 40px; }}
  .tbl {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  .tbl th, .tbl td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: center; font-size: 14px; }}
  .tbl th {{ background: #f2f2f2; }}
  img {{ max-width: 100%; margin-top: 12px; }}
  .meta {{ color: #777; font-size: 13px; }}
</style>
</head>
<body>
  <h1>할리스 매장 밀도 분석 리포트</h1>
  <p class="meta">생성 시각: {now}</p>

  <h2>1. 시도별 인구 대비 매장 수 (요약표)</h2>
  {table_html}

  <h2>2. 막대그래프: 인구 10만명당 매장 수</h2>
  {f'<img src="data:image/png;base64,{bar_b64}">' if bar_b64 else '<p>이미지 없음</p>'}

  <h2>3. 산점도: 인구 vs 매장 수</h2>
  {f'<img src="data:image/png;base64,{scatter_b64}">' if scatter_b64 else '<p>이미지 없음</p>'}

  <h2>4. 시도별 매장 밀도 지도 (Choropleth)</h2>
  {map_iframe if map_iframe else '<p>지도 없음 (source/skorea-provinces-2018-geo.json 확인 필요)</p>'}

</body>
</html>
"""
    path = f"{OUTPUT_DIR}/report.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print("최종 리포트 저장 완료:", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="할리스 매장 밀도 분석 파이프라인")
    parser.add_argument("--skip-crawl", action="store_true", help="크롤링 생략 (source/hollys_store.csv 필요)")
    parser.add_argument("--skip-geocode", action="store_true", help="좌표변환 생략 (source/hollys_store_geo_kakao_final.csv 필요)")
    args = parser.parse_args()

    ensure_dirs()

    if not args.skip_crawl:
        step1_crawl()
    else:
        print("[1단계 생략] source/hollys_store.csv 사용")

    if not args.skip_geocode:
        step2_geocode()
    else:
        print("[2단계 생략] source/hollys_store_geo_kakao_final.csv 사용")

    store_count = step3_store_count()
    df_pop = step4_population()
    df_merge = step5_merge(store_count, df_pop)
    step_cluster()

    df_report = step6_report(df_merge)
    bar_png = step7_barplot(df_merge)
    scatter_png = step7b_scatter(df_merge)
    map_html = step8_map(df_merge)

    build_html_report(df_report, bar_png, scatter_png, map_html)
    print("\n✅ 전체 파이프라인 완료. output/report.html 을 브라우저로 열어 확인하세요.")


if __name__ == "__main__":
    main()