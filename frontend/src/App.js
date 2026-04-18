import React, { useMemo, useState } from "react";
import "./App.css";

const API_BASE = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

function pct(x) {
  return `${(x * 100).toFixed(2)}%`;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const probsSorted = useMemo(() => {
    if (!result?.probabilities) return [];
    return Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  }, [result]);

  const onFileChange = (e) => {
    const f = e.target.files?.[0];
    setFile(f || null);
    setResult(null);
    setError("");

    if (f) setPreview(URL.createObjectURL(f));
    else setPreview(null);
  };

  const onPredict = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "API error");
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const confidence = result?.confidence ?? 0;
  const lowConfidence = result && confidence < 0.6;

  return (
    <div className="container">
      <div className="header">
        <h1 className="title">Eye Disease Detection (ML)</h1>
        <p className="subtitle">
          Upload an eye image to detect: <b>cataract</b>, <b>diabetic retinopathy</b>, <b>glaucoma</b>, or <b>normal</b>.
        </p>
      </div>

      <div className="grid">
        {/* Upload */}
        <div className="card">
          <h2>Upload Image</h2>

          <div className="row" style={{ justifyContent: "space-between", flexWrap: "wrap" }}>
            <input type="file" accept="image/png,image/jpeg" onChange={onFileChange} />
            <button className="btn" onClick={onPredict} disabled={!file || loading}>
              {loading && <span className="spinner" />}
              {loading ? "Predicting…" : "Predict"}
            </button>
          </div>

          {preview && (
            <div style={{ marginTop: 14 }}>
              <div className="label" style={{ marginBottom: 8 }}>Original</div>
              <img src={preview} alt="preview" className="image" />
            </div>
          )}

          {error && (
            <div className="badge danger">
              <b>Error:</b> {error}
            </div>
          )}
        </div>

        {/* Results */}
        <div className="card">
          <h2>Prediction Result</h2>

          {!result && <p className="muted">Upload an image and click Predict.</p>}

          {result && (
            <>
              <div className="kpi">
                <div className="kpiBox">
                  <div className="muted">Prediction</div>
                  <div className="kpiValue">{result.predicted_class}</div>
                </div>
                <div className="kpiBox">
                  <div className="muted">Confidence</div>
                  <div className="kpiValue">{pct(confidence)}</div>
                </div>
              </div>

              {lowConfidence && (
                <div className="badge warn">
                  ⚠️ Confidence is low. The image may be unclear or the case may be difficult. Treat this as a tentative output.
                </div>
              )}

              <div className="sectionTitle">Class Probabilities</div>
              <div>
                {probsSorted.map(([label, p]) => (
                  <div key={label} className="probRow">
                    <div>{label}</div>
                    <div className="bar">
                      <div className={`fill ${label}`} style={{ width: `${p * 100}%` }} />
                    </div>
                    <div style={{ textAlign: "right" }}>{pct(p)}</div>
                  </div>
                ))}
              </div>

              <div className="sectionTitle">Grad-CAM Heatmap</div>
              <p className="muted" style={{ marginTop: 0 }}>
                Highlighted regions contributed most to the model’s prediction (explainability).
              </p>

              {result.gradcam_overlay_base64 && (
                <img
                  src={`data:image/png;base64,${result.gradcam_overlay_base64}`}
                  alt="gradcam"
                  className="image"
                />
              )}
            </>
          )}
        </div>
      </div>

      {/* Medical Info */}
      <div className="card footerCard" style={{ marginTop: 18 }}>
        <h2>Medical Information</h2>

        <p className="muted">
          <b>Disclaimer:</b> This application is an educational/research prototype. It is <b>not</b> a certified medical device and must
          not be used as a substitute for professional diagnosis.
        </p>

        <ul>
          <li><b>Cataract:</b> Clouding of the eye lens causing blurry vision and reduced contrast.</li>
          <li><b>Diabetic Retinopathy:</b> Damage to retinal blood vessels due to diabetes; may lead to vision loss.</li>
          <li><b>Glaucoma:</b> Optic nerve damage, often related to increased intraocular pressure.</li>
          <li><b>Normal:</b> No strong disease patterns detected by the model.</li>
        </ul>

        <p className="muted">
          <b>Interpretation:</b> Confidence shows model certainty. Grad-CAM is an explanation tool showing influential regions.
        </p>
      </div>
    </div>
  );
}
