import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="Cardio Risk Classifier", layout="wide")
st.title("🚑 Cardio Health Risk Predictor")
st.write("Upload file CSV có cột 'cardio' và chạy Random Forest.")

uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
    else:
        st.subheader("📊 5 dòng đầu:")
        st.write(df.head())

        if 'cardio' not in df.columns:
            st.error("⚠️ File CSV thiếu cột 'cardio'!")
        else:
            features = df.drop(columns=[col for col in ["id", "cardio"] if col in df.columns])
            target = df["cardio"]

            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

            st.info("🔍 Đang huấn luyện mô hình Random Forest...")

            param_grid = {
                "n_estimators": [100],
                "max_depth": [None],
                "min_samples_split": [2],
                "min_samples_leaf": [1]
            }

            model = RandomForestClassifier(random_state=42)
            grid = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=-1)
            grid.fit(X_train, y_train)

            y_pred = grid.predict(X_test)

            st.subheader("✅ Kết quả")
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
            st.write(f"**Precision:** {precision_score(y_test, y_pred):.2f}")
            st.write(f"**Recall:** {recall_score(y_test, y_pred):.2f}")
            st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.2f}")
            st.write(f"**AUC:** {roc_auc_score(y_test, grid.predict_proba(X_test)[:,1]):.2f}")
            st.write("**Best Params:**", grid.best_params_)

            st.success("🎉 Huấn luyện thành công! Đã sẵn sàng chạy trên Streamlit Cloud.")
else:
    st.warning("⚠️ Chưa có file CSV, hãy upload!")
