import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta, datetime
import time

# -------------------------- 生成历史空气质量数据集（9列原始数据） --------------------------
def generate_air_data():
    start_date = date(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "重庆"]
    data = []
    for d in dates:
        for city in cities:
            season_factor = 1 + 0.3 * np.sin(2 * np.pi * (d.timetuple().tm_yday / 365))
            pm25 = round(np.random.normal(55, 20) * season_factor, 1)
            pm10 = round(pm25 * 1.5 + np.random.normal(10, 5), 1)
            o3 = round(np.random.normal(85, 25) * (1 - 0.2 * season_factor), 1)
            no2 = round(np.random.normal(35, 12), 1)
            so2 = round(np.random.normal(12, 5), 1)
            temp = round(15 + 20 * np.sin(2 * np.pi * (d.timetuple().tm_yday / 365 - 0.25)), 1)
            aqi = round(pm25 * 1.2 + o3 * 0.4, 0)
            if np.random.random() < 0.05:
                pm25 = np.nan
            if np.random.random() < 0.03:
                o3 = np.nan
            data.append([d, city, pm25, pm10, o3, no2, so2, temp, aqi])
    df = pd.DataFrame(data, columns=["date", "city", "PM2_5", "PM10", "O3", "NO2", "SO2", "temperature", "AQI"])
    df.to_csv("air_quality.csv", index=False, encoding="utf-8-sig")
    return df

# -------------------------- 实时模拟监测数据流【修复列数不匹配bug】 --------------------------
def get_realtime_data(old_df_raw):
    cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "重庆"]
    now = datetime.now()
    new_rows = []
    # 只生成原始9个基础字段，预处理函数会自动补充year/month/season/aqi_level
    for city in cities:
        pm25 = round(np.random.normal(50, 15), 1)
        pm10 = round(pm25 * 1.4 + np.random.normal(8, 4), 1)
        o3 = round(np.random.normal(75, 20), 1)
        no2 = round(np.random.normal(30, 10), 1)
        so2 = round(np.random.normal(10, 4), 1)
        temp = round(np.random.normal(22, 6), 1)
        aqi = round(pm25 * 1.2 + o3 * 0.4, 0)
        # 严格9个字段，和csv原始列一一对应
        new_rows.append([now.date(), city, pm25, pm10, o3, no2, so2, temp, aqi])
    # 只用原始9列创建新数据，不要用预处理后的13列old_df
    raw_cols = ["date", "city", "PM2_5", "PM10", "O3", "NO2", "SO2", "temperature", "AQI"]
    new_df = pd.DataFrame(new_rows, columns=raw_cols)
    # 合并原始历史数据（只取原始9列，避免衍生列冲突）
    old_raw = old_df_raw[raw_cols].copy()
    combine_raw = pd.concat([old_raw, new_df], ignore_index=True)
    # 合并后再统一做预处理，补齐衍生列
    combine_df = preprocess_raw_df(combine_raw)
    return combine_df

# -------------------------- 单独原始数据预处理函数（统一处理9列原始数据，生成13列完整数据） --------------------------
def preprocess_raw_df(df):
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map({1:"冬季",2:"冬季",3:"春季",4:"春季",5:"春季",
                                     6:"夏季",7:"夏季",8:"夏季",9:"秋季",10:"秋季",
                                     11:"秋季",12:"冬季"})
    numeric_cols = ["PM2_5", "PM10", "O3", "NO2", "SO2", "temperature", "AQI"]
    df[numeric_cols] = df.groupby("city")[numeric_cols].transform(
        lambda x: x.interpolate(method="linear").ffill().bfill()
    )
    def get_aqi_level(aqi):
        if aqi <= 50:
            return "优"
        elif aqi <= 100:
            return "良"
        elif aqi <= 150:
            return "轻度污染"
        elif aqi <= 200:
            return "中度污染"
        elif aqi <= 300:
            return "重度污染"
        else:
            return "严重污染"
    df["aqi_level"] = df["AQI"].apply(get_aqi_level)
    df[numeric_cols] = df[numeric_cols].clip(lower=0)
    return df

# -------------------------- 加载入口函数 --------------------------
def load_and_preprocess(file_path="air_quality.csv"):
    try:
        raw_df = pd.read_csv(file_path, encoding="utf-8-sig")
    except FileNotFoundError:
        raw_df = generate_air_data()
    full_df = preprocess_raw_df(raw_df)
    return full_df

# -------------------------- 绘图函数（修复时序color列名bug） --------------------------
def plot_time_series(df, cities, pollutants, start_date, end_date):
    mask = (df["city"].isin(cities)) & (df["date"] >= start_date) & (df["date"] <= end_date)
    plot_df = df[mask].melt(
        id_vars=["date", "city"], 
        value_vars=pollutants, 
        var_name="污染物", 
        value_name="浓度"
    )
    fig = px.line(
        plot_df, 
        x="date", 
        y="浓度", 
        color="city", 
        facet_col="污染物", 
        facet_col_wrap=2,
        labels={"city":"城市"},
        title=f"{','.join(cities)} 污染物浓度时序变化"
    )
    fig.update_layout(height=600, hovermode="x unified")
    return fig

def plot_city_heatmap(df, time_range="全年"):
    pivot_df = df.groupby("city")[["PM2_5", "PM10", "O3", "NO2", "SO2", "AQI"]].mean().round(1)
    fig = px.imshow(pivot_df, text_auto=True, color_continuous_scale="RdYlGn_r",
                    title=f"{time_range} 各城市污染物平均浓度热力图")
    fig.update_layout(height=500, xaxis_title="污染物", yaxis_title="城市")
    return fig

def plot_corr_heatmap(df):
    corr_df = df[["PM2_5", "PM10", "O3", "NO2", "SO2", "temperature", "AQI"]].corr().round(3)
    fig = px.imshow(corr_df, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="污染物与气象因子相关性矩阵")
    fig.update_layout(height=550)
    return fig

def plot_level_pie(df):
    level_counts = df["aqi_level"].value_counts().reset_index()
    level_counts.columns = ["空气质量等级", "天数"]
    color_map = {"优":"#2E8B57", "良":"#90EE90", "轻度污染":"#FFD700",
                 "中度污染":"#FF8C00", "重度污染":"#DC143C", "严重污染":"#8B0000"}
    fig = px.pie(level_counts, names="空气质量等级", values="天数", color="空气质量等级",
                 color_discrete_map=color_map, hole=0.4, title="空气质量等级天数占比")
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=500)
    return fig

# -------------------------- Streamlit主程序 --------------------------
def main():
    st.set_page_config(page_title="城市空气质量可视化平台", layout="wide")
    st.sidebar.title("导航菜单")
    page = st.sidebar.selectbox("请选择功能页面：",
                                ["数据总览", "时序趋势分析", "区域污染对比", "相关性分析", "统计看板", "🔴 实时空气质量监测"])

    @st.cache_data
    def get_data():
        return load_and_preprocess("air_quality.csv")

    df = get_data()

    # 页面1：数据总览
    if page == "数据总览":
        st.header("📊 空气质量数据总览")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("覆盖城市数量", f"{df['city'].nunique()} 个")
        with col2:
            avg_aqi = round(df["AQI"].mean(), 1)
            st.metric("平均AQI", avg_aqi)
        with col3:
            good_rate = round(len(df[df["aqi_level"].isin(["优","良"])]) / len(df) * 100, 1)
            st.metric("优良天数占比", f"{good_rate} %")
        with col4:
            st.metric("数据时间跨度", f"{df['date'].dt.year.nunique()} 年")
        st.markdown("---")
        st.subheader("原始数据预览")
        city_filter = st.multiselect("筛选城市：", options=df["city"].unique(), default=df["city"].unique()[:3])
        show_df = df[df["city"].isin(city_filter)] if city_filter else df
        st.dataframe(show_df.sort_values("date", ascending=False), use_container_width=True, height=400)

    # 页面2：时序趋势分析
    elif page == "时序趋势分析":
        st.header("📈 污染物时序趋势分析")
        col1, col2 = st.columns(2)
        with col1:
            selected_cities = st.multiselect("选择城市：", options=df["city"].unique(), default=["北京", "上海"])
        with col2:
            selected_pollutants = st.multiselect("选择污染物：",
                                                options=["PM2_5", "PM10", "O3", "NO2", "SO2", "AQI"],
                                                default=["PM2_5", "O3"])
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        start_d, end_d = st.slider("选择时间范围：", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        st.markdown("---")
        if selected_cities and selected_pollutants:
            fig = plot_time_series(df, selected_cities, selected_pollutants, pd.Timestamp(start_d), pd.Timestamp(end_d))
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 分析要点：观察冬季与夏季的浓度差异、节假日前后的波动、不同城市的趋势同步性")
        else:
            st.warning("请至少选择1个城市和1种污染物")

    # 页面3：区域污染对比
    elif page == "区域污染对比":
        st.header("🗺️ 区域污染水平对比")
        time_dim = st.radio("统计维度：", ["全年平均", "分季节", "分月份"], horizontal=True)
        if time_dim == "全年平均":
            fig = plot_city_heatmap(df, "全年")
        elif time_dim == "分季节":
            season = st.selectbox("选择季节：", df["season"].unique())
            fig = plot_city_heatmap(df[df["season"]==season], season)
        else:
            month = st.slider("选择月份：", 1, 12, 1)
            fig = plot_city_heatmap(df[df["month"]==month], f"{month}月")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 城市AQI排名")
        rank_df = df.groupby("city")["AQI"].mean().sort_values(ascending=False).round(1).reset_index()
        rank_df.columns = ["城市", "平均AQI"]
        st.dataframe(rank_df, use_container_width=True)

    # 页面4：相关性分析
    elif page == "相关性分析":
        st.header("🔗 污染物相关性分析")
        city_corr = st.selectbox("选择分析城市：", options=["全部城市"] + list(df["city"].unique()))
        data_corr = df if city_corr == "全部城市" else df[df["city"] == city_corr]
        fig = plot_corr_heatmap(data_corr)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 结果解读")
        st.write("1. PM2.5与PM10通常呈强正相关，来源具有同源性（如扬尘、燃煤）")
        st.write("2. 臭氧与温度呈正相关，夏季高温时光化学反应强烈，臭氧浓度升高")
        st.write("3. AQI与PM2.5相关性最高，说明PM2.5是影响空气质量的首要污染物")

    # 页面5：统计看板
    elif page == "统计看板":
        st.header("📋 空气质量综合统计看板")
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = plot_level_pie(df)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            month_aqi = df.groupby("month")["AQI"].mean().reset_index()
            fig_bar = px.bar(month_aqi, x="month", y="AQI", title="月度平均AQI变化",
                             color="AQI", color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("---")
        st.subheader("各城市详细统计指标")
        city_stats = df.groupby("city").agg(
            平均AQI=("AQI","mean"),
            最高AQI=("AQI","max"),
            优良天数占比=("aqi_level", lambda x: round(sum(x.isin(["优","良"]))/len(x)*100,1)),
            重度及以上天数=("aqi_level", lambda x: sum(x.isin(["重度污染","严重污染"])))
        ).round(1).sort_values("平均AQI", ascending=False)
        st.dataframe(city_stats, use_container_width=True)

    # 页面6：实时监测面板（已修复列数冲突）
    elif page == "🔴 实时空气质量监测":
        st.header("🔴 城市空气质量实时监测面板")
        refresh_interval = st.slider("刷新间隔(秒)", min_value=2, max_value=10, value=3)
        chart_placeholder = st.empty()
        table_placeholder = st.empty()
        info_text = st.empty()

        while True:
            # 传入原始完整历史数据，函数内部自动处理列匹配
            live_df = get_realtime_data(df)
            # 实时AQI柱状图
            live_avg = live_df.groupby("city")["AQI"].mean().reset_index()
            fig_live = px.bar(live_avg, x="city", y="AQI", title="各城市实时AQI值",
                              color="AQI", color_continuous_scale="RdYlGn_r", labels={"city":"城市"})
            chart_placeholder.plotly_chart(fig_live, use_container_width=True)
            # 最新实时数据表格
            table_placeholder.dataframe(live_df.tail(10), use_container_width=True)
            # 刷新时间提示
            now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info_text.info(f"✅ 数据已刷新，当前时间：{now_time} | 每{refresh_interval}秒自动更新")
            time.sleep(refresh_interval)

if __name__ == "__main__":
    main()