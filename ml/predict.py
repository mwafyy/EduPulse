import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "ml", "saved_models")

_rf_binary = None
_rf_multi = None

def get_binary_model():
    global _rf_binary
    if _rf_binary is None:
        path = os.path.join(MODELS_DIR, "rf_binary_pipeline.joblib")
        _rf_binary = joblib.load(path)
    return _rf_binary

def get_multi_model():
    global _rf_multi
    if _rf_multi is None:
        path = os.path.join(MODELS_DIR, "rf_multiclass_pipeline.joblib")
        _rf_multi = joblib.load(path)
    return _rf_multi

def generate_recommendations(features: dict, risk_level: str, at_risk_prob: float):
    recommendations = []

    # Analyze key indicators
    studytime = features.get('studytime', 2)
    absences = features.get('absences', 0)
    failures = features.get('failures', 0)
    schoolsup = features.get('schoolsup', 'no')
    famsup = features.get('famsup', 'no')
    higher = features.get('higher', 'yes')
    goout = features.get('goout', 3)

    if risk_level in ['High Risk', 'Medium Risk'] or at_risk_prob >= 0.4:
        # Absence alert
        if absences > 8:
            recommendations.append({
                "category": "متابعة الحضور (Attendance Tracking)",
                "action": f"عدد أيام الغياب ({absences} يوم) مرتفع جدًا؛ يُوصى بجدولة مقابلة عاجلة مع ولي الأمر وتفعيل تنبيهات الغياب الأسبوعية.",
                "priority": "High"
            })
        elif absences > 4:
            recommendations.append({
                "category": "متابعة الحضور (Attendance Tracking)",
                "action": f"لوحظ تذبذب في الحضور ({absences} أيام غياب)؛ يجب التنبيه على ضرورة الالتزام بالحضور الصباحي.",
                "priority": "Medium"
            })

        # Study time alert
        if studytime <= 1:
            recommendations.append({
                "category": "خطة الاستذكار (Study Schedule)",
                "action": "وقت المذاكرة أقل من ساعتين أسبوعيًا؛ يجب وضع جدول منظم وتحديد أوقات مراجعة إجبارية في المدرسة/المنزل.",
                "priority": "High"
            })
        elif studytime == 2:
            recommendations.append({
                "category": "خطة الاستذكار (Study Schedule)",
                "action": "وقت الاستذكار الحالي (2-5 ساعات أسبوعيًا) قد لا يكفي في المواد الصعبة؛ يُنصح بزيادة ساعة واحدة للمراجعة المركزة.",
                "priority": "Medium"
            })

        # Past failures alert
        if failures > 0:
            recommendations.append({
                "category": "الدعم الأكاديمي الاستدراكي (Remedial Tutoring)",
                "action": f"الطالب لديه رسوب سابق في ({failures}) مقررات سابقة؛ يوصى بإلحاقه بحصص التقوية وتوفير معلم مرافق لمراجعة الأساسيات.",
                "priority": "Critical"
            })

        # Support systems
        if schoolsup == 'no':
            recommendations.append({
                "category": "الدعم المدرسي الإضافي (School Support)",
                "action": "الطالب لا يستفيد حالياً من الدعم الإضافي بالمدرسة؛ يُقترح إدراجه رسميًا في برامج الدعم المدرسي التكميلي.",
                "priority": "Medium"
            })

        if goout >= 4 and studytime <= 2:
            recommendations.append({
                "category": "إدارة الوقت والتوازن (Time Management)",
                "action": "معدل الخروج مع الأصدقاء مرتفع مقارنة بساعات المذاكرة؛ يُفضل توجيه الطالب حول التوازن بين الترفيه والتحصيل.",
                "priority": "Low"
            })
    else:
        recommendations.append({
            "category": "تعزيز المستوى (Academic Enrichment)",
            "action": "أداء الطالب مستقر؛ يُنصح بتحفيزه على المشاركة في الأنشطة المتقدمة والمسابقات العلمية.",
            "priority": "Info"
        })

    return recommendations

def predict_single(data: dict):
    model_bin = get_binary_model()
    model_multi = get_multi_model()

    df = pd.DataFrame([data])

    binary_pred = int(model_bin.predict(df)[0])
    binary_proba = float(model_bin.predict_proba(df)[0][1])

    multi_pred = str(model_multi.predict(df)[0])
    multi_probas = model_multi.predict_proba(df)[0]
    multi_classes = model_multi.classes_.tolist()

    proba_breakdown = {cls: round(float(prob), 4) for cls, prob in zip(multi_classes, multi_probas)}

    # Calibrated risk level
    if binary_proba >= 0.55:
        risk_level = "High Risk"
    elif binary_proba >= 0.35:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    recs = generate_recommendations(data, risk_level, binary_proba)

    return {
        "at_risk": bool(binary_pred),
        "risk_probability": round(binary_proba * 100, 1),
        "risk_level": risk_level,
        "multiclass_prediction": multi_pred,
        "probabilities": proba_breakdown,
        "recommendations": recs
    }
