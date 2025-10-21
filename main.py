import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib
import streamlit as st
import tensorflow as tf

df= pd.read_csv('CloudWatch_Traffic_Web_Attack.csv')

st.set_page_config(page_title='Web Traffic Dashboard',layout='centered')
st.title('Web Traffic Anamoly Detection Dashboard')
st.subheader("Behavioral Insights for web traffic(waf_rule)")

insights,prediction=st.tabs(['Data Insights','Prediction'])

with insights:
    st.subheader('Data Insights')
    st.markdown('Key Metrics')
    
    def metric_card(label, value):
        st.markdown(f"""
        <div style="
            backdrop-filter: blur(12px) saturate(180%);
            -webkit-backdrop-filter: blur(12px) saturate(180%);
            background-color: rgba(173, 216, 230, 0.25);  /* Light blue tint */
            border: 1px solid rgba(30, 144, 255, 0.3);     /* DodgerBlue border */
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 50, 0.1);
            min-width: 200px;
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin: 10px 8px;
            color: #EAF6FF;
            font-weight: 600;
            text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.3);
        ">
            <div style="font-size:14px;">{label}</div>
            <div style="font-size:18px;">{value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    kpi1,kpi2,kpi3=st.columns(3)
    with kpi1:
        metric_card("Total Records",df.shape[0])
    with kpi2:
        traffic_by_country=df.groupby('src_ip_country_code')[['bytes_in','bytes_out']].sum()
        traffic_by_country['total_traffic']=traffic_by_country['bytes_in']+traffic_by_country['bytes_out']
        top_traffic_by_country=traffic_by_country['total_traffic'].idxmax()
        metric_card('Top Country by Traffic',top_traffic_by_country)
    with kpi3:
        traffic_by_ip=df.groupby('src_ip')[['bytes_in','bytes_out']].sum()
        traffic_by_ip['total_traffic']=traffic_by_ip['bytes_in']+traffic_by_ip['bytes_out']
        top_traffic_ip=traffic_by_ip['total_traffic'].idxmax()
        metric_card("Top IP by Traffic",top_traffic_ip)
    kpi4,kpi5,kpi6=st.columns(3)
    with kpi4:
        df['hour']=pd.to_datetime(df['time']).dt.hour
        peak_hour=df['hour'].value_counts().idxmax()
        metric_card("Peak Traffic Hour",f'{peak_hour}:00')
    with kpi5:
        df['end_time'] = pd.to_datetime(df['end_time'])
        df['creation_time'] = pd.to_datetime(df['creation_time'])
        df['duration'] = (df['end_time'] - df['creation_time']).dt.total_seconds()
        df['total_volume'] = df['bytes_in'] + df['bytes_out']
        avg_volume = (df['total_volume'] / df['duration']).mean()
        metric_card("Avg Volume per Session", f"{avg_volume:,.2f} bytes/sec")
    with kpi6:
        df['traffic_ratio'] = df['bytes_in'] / (df['bytes_out'] + 1e-6)
        avg_ratio = df['traffic_ratio'].mean()
        metric_card("Avg Traffic Ratio", f"{avg_ratio:.2f}")

    st.markdown(f"**◉ Dataset shape : `{df.shape[0]}`rows ✕ `{df.shape[1]}`columns**")
    st.write("column types:")
    st.write(df.dtypes.value_counts())
    st.markdown("◉ Data Description : ")
    st.table(df.describe().T)
    st.markdown("<h3 style='text-align: center;'> Numerical Features Visulaization</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left;'> Distribution & Boxplots</h3>", unsafe_allow_html=True)
    
    numeric_col=['bytes_in','bytes_out']
    for col in numeric_col:
        col1,col2=st.columns(2)
        with col1:
            st.markdown(f"**◉ Distribution of {col}**")
            fig1,ax=plt.subplots(figsize=(5,6))
            sns.histplot(df[col],kde=True,bins=30,color='skyblue',ax=ax)
            ax.axvline(df[col].mean(),color='green',linestyle='--',label='Mean')
            ax.axvline(df[col].median(),color='red',linestyle='-.',label='Median')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig1)
        with col2:
            st.markdown(f"**◉ Boxplot of {col}**")
            fig2,ax2=plt.subplots(figsize=(5,6))
            sns.boxplot(y=df[col],ax=ax2,color="#90EE90")
            ax2.axhline(df[col].mean(), color='green', linestyle='--', label='Mean')
            ax2.axhline(df[col].median(), color='red', linestyle='-.', label='Median')
            ax2.grid(True)
            st.pyplot(fig2)
            
    st.markdown("<h3 style='text-align: center;'>Categorical Features Visulaization</h3>", unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        st.markdown("**◉ Distribution Countries by Traffic**")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.countplot(x='src_ip_country_code',data=df,order=df['src_ip_country_code'].value_counts().index)
        ax.set_xlabel("Country")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
    with col2:
        st.markdown("**◉ Distribution top 10 IPs by Traffic**")
        top_ips=df.value_counts('src_ip').head(10)
        fig, ax = plt.subplots(figsize=(6, 7))
        sns.barplot(y=top_ips.index,x=top_ips.values)
        ax.set_xlabel("Count")
        ax.set_ylabel("IP Address")
        st.pyplot(fig)
    
    st.markdown("<h3 style='text-align: left;'>Traffic Analysis Over Time</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(18, 8))
    sns.lineplot(x='time', y=df['bytes_in'], data=df, label='Bytes In', color='skyblue', marker='o', linestyle='-', ax=ax)
    sns.lineplot(x='time', y=df['bytes_out'], data=df, label='Bytes Out', color='orange', marker='o', linestyle='--', ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Bytes')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)
    fig, ax = plt.subplots(figsize=(18, 8))
    plt.plot(df['time'], df['bytes_in'], label='Bytes In', color='skyblue', marker='o', linestyle='-')
    plt.plot(df['time'], df['bytes_out'], label='Bytes Out', color='orange', marker='s', linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Bytes')
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("<h3 style='text-align: left;'>Relation of Countries over Time</h3>", unsafe_allow_html=True)
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    heatmap_data = df.groupby(['src_ip_country_code', 'hour'])['bytes_in'].sum().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=0.5, annot=True, ax=ax)
    ax.set_title('Bytes In by Country and Hour')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Country Code')
    st.pyplot(fig)

with prediction:
    st.subheader("Anomaly Prediction")
    st.markdown("Submit traffic details to check for WAF_rule anomaly detection.")

    pt = joblib.load('model_resources/power_transformer.pkl')
    iso_model = joblib.load('model_resources/iso_model.pkl')
    svm_model = joblib.load('model_resources/svm_model.pkl')
    nn_model = tf.keras.models.load_model('model_resources/anomaly_nn_model.h5')

    with st.form("prediction_form"):
        bytes_in = st.number_input("Bytes In", min_value=0)
        bytes_out = st.number_input("Bytes Out", min_value=0)
        duration = st.number_input("Session Duration (sec)", min_value=1)
        time_str = st.text_input("Time of Creation(HH:MM:SS)", value="00:00:00")

        submitted = st.form_submit_button("Submit")

    if submitted:
        try:
            h, m, s = map(int, time_str.strip().split(":"))
            time_in_seconds = h * 3600 + m * 60 + s
            
            traffic_ratio = bytes_in / (bytes_out + 1e-6)
            total_per_second = (bytes_in + bytes_out) / duration

            input_raw = np.array([[bytes_in, bytes_out, duration, traffic_ratio, total_per_second, time_in_seconds]])
            input_scaled = pt.transform(input_raw)

            iso_flag = int(iso_model.predict(input_scaled)[0] == -1)
            svm_flag = int(svm_model.predict(input_scaled)[0] == -1)
            nn_prob = nn_model.predict(input_scaled)[0][0]
            nn_flag = int(nn_prob > 0.45)

            votes = iso_flag + svm_flag + nn_flag
            ensemble_flag = int(votes >= 2)

            if ensemble_flag == 1:
                st.success("Majority Vote: Not Recognized as WAF-rule Suspicion (Normal Traffic)")
            else:
                st.info("Majority Vote: Recognized as WAF-rule Suspicion (Anomaly Detected)")

            st.markdown(f"""
            **Model Votes:**  
            - Isolation Forest: `{iso_flag}`
            - One-Class SVM: `{svm_flag}`
            - Neural Net: `{nn_flag}`
            - Ensemble Total: `{votes}`
            """)

        except Exception as e:
            st.error(f"Invalid time format. Please use HH:MM:SS. Error: {e}")
    st.markdown("### Model Training Summary")
    st.markdown("**Model flagged `10` out of `282` suspicious entries as anomalous.**")
    st.markdown("**Threshold used:** `0.45`")


    
