"""
MEDIRISK - Prototype CMMS & Decision Support for Medical Equipment
Run:
    pip install streamlit pandas numpy
    streamlit run medirisk_cmms.py

IMPORTANT:
- This is a research/demo prototype, not a certified calibration system.
- Configure limits only from approved manufacturer procedures, applicable standards,
  and validated local SOPs. Example tolerances are intentionally marked as demo values.
- System recommendations do not replace inspection, calibration, or decisions by authorized personnel.
"""
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "medirisk.db"

st.set_page_config(page_title="MEDIRISK | CMMS Elektromedis", page_icon="🏥", layout="wide")

# Demo defaults only. Replace/validate these values against the exact device parameter,
# range, manufacturer specification, test procedure, and applicable standards.
DEVICE_TYPES = [
    "Suction Pump", "Infusion Pump", "Syringe Pump", "Patient Monitor",
    "Ventilator", "Defibrillator", "Infant Warmer", "Other"
]
FUNCTION_SCORES = {
    "Life support": 10, "Terapi/bedah/intensif": 8, "Monitoring/diagnostik": 6,
    "Laboratorium": 4, "Pendukung": 2
}
PHYSICAL_RISK_SCORES = {
    "Potensi fatal": 5, "Cedera berat": 4, "Cedera ringan": 3,
    "Kerusakan alat saja": 2, "Risiko minimal": 1
}
MAINTENANCE_SCORES = {
    "Sangat kompleks": 5, "Kompleks": 4, "Standar": 3,
    "Sederhana": 2, "Inspeksi visual": 1
}

# These are clearly labelled as prototype/demo settings, not universal standards.
DEMO_LIMITS = {
    "Suction Pump": {"parameter": "Tekanan/vakum (sesuaikan SOP)", "unit": "mmHg", "limit_pct": 5.0},
    "Infusion Pump": {"parameter": "Laju alir (sesuaikan SOP)", "unit": "mL/h", "limit_pct": 5.0},
    "Syringe Pump": {"parameter": "Laju alir (sesuaikan SOP)", "unit": "mL/h", "limit_pct": 2.0},
    "Patient Monitor": {"parameter": "Parameter simulasi (sesuaikan SOP)", "unit": "bpm", "limit_pct": 2.0},
    "Ventilator": {"parameter": "Parameter simulasi (sesuaikan SOP)", "unit": "sesuai SOP", "limit_pct": 5.0},
    "Defibrillator": {"parameter": "Parameter simulasi (sesuaikan SOP)", "unit": "sesuai SOP", "limit_pct": 5.0},
    "Infant Warmer": {"parameter": "Parameter simulasi (sesuaikan SOP)", "unit": "°C", "limit_pct": 2.0},
    "Other": {"parameter": "Parameter alat", "unit": "sesuai SOP", "limit_pct": 5.0},
}

def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS assets (
            asset_id TEXT PRIMARY KEY,
            device_type TEXT NOT NULL,
            room TEXT DEFAULT '',
            manufacturer TEXT DEFAULT '',
            model TEXT DEFAULT '',
            serial_no TEXT DEFAULT '',
            function_score INTEGER NOT NULL,
            physical_risk_score INTEGER NOT NULL,
            maintenance_score INTEGER NOT NULL,
            incident_score INTEGER NOT NULL DEFAULT 0,
            operating_hours REAL,
            standard_hours REAL,
            mtbf_reference REAL,
            mtbf_actual REAL,
            deviation_pct REAL,
            deviation_limit_pct REAL,
            last_calibration TEXT,
            calibration_due TEXT,
            last_pm TEXT,
            status TEXT NOT NULL DEFAULT 'Perlu peninjauan',
            created_at TEXT NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT NOT NULL,
            test_date TEXT NOT NULL,
            parameter_name TEXT NOT NULL,
            unit TEXT NOT NULL,
            reference_value REAL NOT NULL,
            readings_json TEXT NOT NULL,
            mean_value REAL NOT NULL,
            error_pct REAL NOT NULL,
            configured_limit_pct REAL NOT NULL,
            result TEXT NOT NULL,
            notes TEXT DEFAULT '',
            FOREIGN KEY(asset_id) REFERENCES assets(asset_id)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS work_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            priority TEXT NOT NULL,
            reason TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            due_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Open',
            FOREIGN KEY(asset_id) REFERENCES assets(asset_id)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            asset_id TEXT,
            event TEXT NOT NULL,
            details TEXT NOT NULL
        )
        """)

def audit(asset_id, event, details):
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO audit_log(created_at, asset_id, event, details) VALUES(?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), asset_id, event, details)
        )

def load_assets():
    with db_conn() as conn:
        return pd.read_sql_query("SELECT * FROM assets ORDER BY asset_id", conn)

def load_tests(asset_id=None):
    with db_conn() as conn:
        if asset_id:
            return pd.read_sql_query(
                "SELECT * FROM calibration_tests WHERE asset_id=? ORDER BY test_date DESC, id DESC",
                conn, params=(asset_id,)
            )
        return pd.read_sql_query("SELECT * FROM calibration_tests ORDER BY test_date DESC, id DESC", conn)

def load_work_orders():
    with db_conn() as conn:
        return pd.read_sql_query("SELECT * FROM work_orders ORDER BY CASE priority WHEN 'Kritis' THEN 1 WHEN 'Tinggi' THEN 2 WHEN 'Sedang' THEN 3 ELSE 4 END, due_date", conn)

def equipment_score(row):
    # EM score is used as a prioritization indicator in this prototype.
    return int(row["function_score"]) + int(row["physical_risk_score"]) + int(row["maintenance_score"]) + int(row["incident_score"])

def calculate_dri(row):
    """
    DRI prototype: weighted sum of four normalized indicators.
    It is a prioritization index, NOT a validated probability of failure.
    A component is required to calculate a complete score.
    """
    required = [
        row.get("operating_hours"), row.get("standard_hours"),
        row.get("mtbf_reference"), row.get("mtbf_actual"),
        row.get("deviation_pct"), row.get("deviation_limit_pct")
    ]
    if any(pd.isna(x) or x is None for x in required):
        return None, ["Data DRI belum lengkap: jam operasi, standar jam, MTBF, dan deviasi harus diisi."]
    oh, sh, mtbf_ref, mtbf_act, dev, dev_lim = [float(x) for x in required]
    if sh <= 0 or mtbf_ref <= 0 or mtbf_act <= 0 or dev_lim <= 0:
        return None, ["Data acuan DRI harus lebih besar dari nol."]
    em = equipment_score(row)
    # Clamp components to prevent one raw ratio from dominating without bound.
    em_component = np.clip(em / 20.0, 0.0, 1.0)
    hours_component = np.clip(oh / sh, 0.0, 2.0) / 2.0
    mtbf_component = np.clip(mtbf_ref / mtbf_act, 0.0, 2.0) / 2.0
    deviation_component = np.clip(abs(dev) / dev_lim, 0.0, 2.0) / 2.0
    dri = (0.35 * em_component + 0.25 * hours_component
           + 0.20 * mtbf_component + 0.20 * deviation_component)
    return float(dri), []

def explain_rules(row, dri):
    """Transparent rules. Safety flags are separate from the aggregate DRI."""
    reasons, actions = [], []
    priority = "Rendah"
    due_days = 30

    due = pd.to_datetime(row.get("calibration_due"), errors="coerce")
    today = pd.Timestamp(date.today())
    if pd.notna(due):
        remaining = (due.normalize() - today).days
        if remaining < 0:
            reasons.append(f"Tenggat kalibrasi terlewat {abs(remaining)} hari.")
            actions.append("Segera verifikasi status sertifikat dan atur tindak lanjut kalibrasi.")
            priority, due_days = "Tinggi", min(due_days, 1)
        elif remaining <= 30:
            reasons.append(f"Kalibrasi jatuh tempo dalam {remaining} hari.")
            actions.append("Jadwalkan/konfirmasi kalibrasi sebelum tenggat wajib.")
            if priority != "Tinggi":
                priority, due_days = "Sedang", min(due_days, max(1, remaining))

    last_test = load_tests(row["asset_id"])
    if not last_test.empty and str(last_test.iloc[0]["result"]).upper() == "FAIL":
        reasons.append("Hasil uji terakhir tercatat FAIL.")
        actions.append("Tinjau hasil oleh petugas berwenang; ikuti SOP untuk pembatasan penggunaan, perbaikan, dan pengujian ulang.")
        priority, due_days = "Kritis", 0

    if pd.notna(row.get("deviation_pct")) and pd.notna(row.get("deviation_limit_pct")):
        try:
            if abs(float(row["deviation_pct"])) > float(row["deviation_limit_pct"]):
                reasons.append("Deviasi yang dicatat melebihi batas konfigurasi.")
                actions.append("Verifikasi hasil ukur dan lakukan evaluasi teknis sesuai SOP alat.")
                priority, due_days = "Tinggi", min(due_days, 1)
        except (TypeError, ValueError):
            pass

    if dri is not None:
        if dri >= 0.75:
            reasons.append(f"DRI {dri:.3f} berada pada kategori prioritas tinggi (ambang prototipe ≥ 0,75).")
            actions.append("Prioritaskan inspeksi/pemeliharaan dan tinjau data pendukung.")
            if priority not in ("Kritis", "Tinggi"):
                priority, due_days = "Tinggi", min(due_days, 7)
        elif dri >= 0.50:
            reasons.append(f"DRI {dri:.3f} berada pada kategori prioritas sedang (ambang prototipe ≥ 0,50).")
            actions.append("Tinjau riwayat dan jadwalkan inspeksi sesuai kapasitas serta SOP.")
            if priority == "Rendah":
                priority, due_days = "Sedang", min(due_days, 14)

    if not reasons:
        reasons.append("Tidak ada pemicu aturan dari data yang tersedia; ini bukan sertifikasi bahwa alat aman.")
        actions.append("Lanjutkan pemantauan dan patuhi jadwal pemeliharaan/kalibrasi yang berlaku.")

    due_date = (date.today() + timedelta(days=due_days)).isoformat()
    return priority, reasons, actions, due_date

init_db()
st.title("🏥 MEDIRISK")
st.caption("Prototipe CMMS elektromedis: inventaris, indeks risiko, rekomendasi berbasis aturan, dan pencatatan uji.")
st.warning(
    "Prototipe riset/demo. DRI dan ambang prioritas adalah parameter awal yang harus divalidasi. "
    "Toleransi uji contoh bukan standar universal. Sistem tidak menetapkan kelayakan alat secara mandiri."
)

tab_dash, tab_assets, tab_risk, tab_test, tab_wo, tab_history = st.tabs([
    "📊 Dashboard", "🗄️ Inventaris", "🧭 Risiko & Rekomendasi",
    "🔧 Catat Uji", "🛠️ Work Order", "🧾 Riwayat & Audit"
])

assets = load_assets()

with tab_dash:
    st.subheader("Ringkasan")
    assets = load_assets()
    wo = load_work_orders()
    tests = load_tests()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total alat", len(assets))
    c2.metric("Work order terbuka", int((wo["status"] == "Open").sum()) if not wo.empty else 0)
    c3.metric("Catatan uji", len(tests))
    c4.metric("Tenggat kalibrasi lewat", int((pd.to_datetime(assets["calibration_due"], errors="coerce").dt.date < date.today()).sum()) if not assets.empty else 0)
    st.divider()
    if not assets.empty:
        display = assets[["asset_id", "device_type", "room", "calibration_due", "status"]].copy()
        display.columns = ["ID Alat", "Jenis", "Ruangan", "Jatuh Tempo Kalibrasi", "Status Catatan"]
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada alat. Tambahkan data melalui tab Inventaris.")
    st.caption("Metrik ini merangkum data yang dicatat dalam aplikasi, bukan bukti peningkatan keselamatan atau penurunan downtime.")

with tab_assets:
    st.subheader("Registrasi / Pembaruan Alat")
    with st.form("asset_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            asset_id = st.text_input("ID / nomor inventaris *").strip()
            device_type = st.selectbox("Jenis alat", DEVICE_TYPES)
            room = st.text_input("Ruangan/unit")
            manufacturer = st.text_input("Pabrikan")
        with col2:
            model = st.text_input("Model")
            serial_no = st.text_input("Nomor seri")
            function_label = st.selectbox("Fungsi klinis", list(FUNCTION_SCORES))
            risk_label = st.selectbox("Risiko fisik", list(PHYSICAL_RISK_SCORES))
        with col3:
            maintenance_label = st.selectbox("Kebutuhan pemeliharaan", list(MAINTENANCE_SCORES))
            incident_score = st.selectbox("Skor riwayat insiden", [-2, -1, 0, 1, 2], index=2,
                                          help="Nilai penilaian internal; definisikan kriteria konsisten sebelum penggunaan.")
            operating_hours = st.number_input("Jam operasi aktual", min_value=0.0, value=0.0, step=10.0)
            standard_hours = st.number_input("Acuan jam operasi", min_value=0.0, value=1000.0, step=10.0)
        st.markdown("**Data untuk DRI (isi berdasarkan catatan yang dapat diverifikasi)**")
        d1, d2, d3 = st.columns(3)
        with d1:
            mtbf_ref = st.number_input("MTBF acuan (jam)", min_value=0.0, value=1000.0, step=10.0)
        with d2:
            mtbf_actual = st.number_input("MTBF aktual (jam)", min_value=0.0, value=1000.0, step=10.0)
        with d3:
            deviation_pct = st.number_input("Deviasi terukur (%)", value=0.0, step=0.1, format="%.3f")
            deviation_limit_pct = st.number_input("Batas deviasi konfigurasi (%)", min_value=0.0, value=5.0, step=0.1)
        e1, e2, e3 = st.columns(3)
        with e1:
            last_calibration = st.date_input("Tanggal kalibrasi terakhir", value=None)
        with e2:
            calibration_due = st.date_input("Tanggal jatuh tempo kalibrasi", value=date.today() + timedelta(days=180))
        with e3:
            last_pm = st.date_input("Tanggal PM terakhir", value=None)
        save = st.form_submit_button("💾 Simpan alat", type="primary", use_container_width=True)

    if save:
        if not asset_id:
            st.error("ID alat wajib diisi.")
        elif standard_hours <= 0 or mtbf_ref <= 0 or mtbf_actual <= 0:
            st.error("Acuan jam operasi dan kedua nilai MTBF harus lebih besar dari nol.")
        elif deviation_limit_pct <= 0:
            st.error("Batas deviasi harus lebih besar dari nol.")
        else:
            em = FUNCTION_SCORES[function_label] + PHYSICAL_RISK_SCORES[risk_label] + MAINTENANCE_SCORES[maintenance_label] + int(incident_score)
            pm_interval = 6 if em >= 12 else 12 if em >= 9 else None
            pm_text = f"Indikasi tinjau interval {pm_interval} bulan (aturan prototipe)" if pm_interval else "Tentukan sesuai SOP/pabrikan"
            with db_conn() as conn:
                conn.execute("""
                INSERT INTO assets(asset_id, device_type, room, manufacturer, model, serial_no,
                    function_score, physical_risk_score, maintenance_score, incident_score,
                    operating_hours, standard_hours, mtbf_reference, mtbf_actual, deviation_pct,
                    deviation_limit_pct, last_calibration, calibration_due, last_pm, status, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    device_type=excluded.device_type, room=excluded.room, manufacturer=excluded.manufacturer,
                    model=excluded.model, serial_no=excluded.serial_no, function_score=excluded.function_score,
                    physical_risk_score=excluded.physical_risk_score, maintenance_score=excluded.maintenance_score,
                    incident_score=excluded.incident_score, operating_hours=excluded.operating_hours,
                    standard_hours=excluded.standard_hours, mtbf_reference=excluded.mtbf_reference,
                    mtbf_actual=excluded.mtbf_actual, deviation_pct=excluded.deviation_pct,
                    deviation_limit_pct=excluded.deviation_limit_pct, last_calibration=excluded.last_calibration,
                    calibration_due=excluded.calibration_due, last_pm=excluded.last_pm, status=excluded.status
                """, (asset_id, device_type, room, manufacturer, model, serial_no,
                      FUNCTION_SCORES[function_label], PHYSICAL_RISK_SCORES[risk_label],
                      MAINTENANCE_SCORES[maintenance_label], int(incident_score),
                      operating_hours, standard_hours, mtbf_ref, mtbf_actual, deviation_pct,
                      deviation_limit_pct, last_calibration.isoformat() if last_calibration else None,
                      calibration_due.isoformat(), last_pm.isoformat() if last_pm else None,
                      "Perlu peninjauan", datetime.now().isoformat(timespec="seconds")))
            audit(asset_id, "UPSERT_ASSET", f"EM Score={em}; {pm_text}")
            st.success(f"Data {asset_id} tersimpan. EM Score = {em}. {pm_text}")
            st.rerun()

    assets = load_assets()
    st.markdown("**Inventaris tersimpan**")
    if not assets.empty:
        st.dataframe(assets, use_container_width=True, hide_index=True)
        del_id = st.selectbox("Pilih ID untuk penghapusan", assets["asset_id"].tolist(), key="del_asset")
        confirm = st.checkbox("Saya memahami penghapusan akan menghilangkan data inventaris ini.", key="confirm_delete")
        if st.button("Hapus inventaris terpilih", disabled=not confirm):
            with db_conn() as conn:
                conn.execute("DELETE FROM assets WHERE asset_id=?", (del_id,))
            audit(del_id, "DELETE_ASSET", "Inventaris dihapus oleh pengguna.")
            st.warning(f"Inventaris {del_id} dihapus. Catatan historis yang sudah tercatat tidak otomatis dihapus.")
            st.rerun()

with tab_risk:
    st.subheader("Dynamic Risk Index dan alasan rekomendasi")
    assets = load_assets()
    if assets.empty:
        st.info("Daftarkan alat terlebih dahulu.")
    else:
        risk_rows = []
        for _, row in assets.iterrows():
            em = equipment_score(row)
            dri, missing = calculate_dri(row)
            priority, reasons, actions, due_date = explain_rules(row, dri)
            risk_rows.append({
                "ID Alat": row["asset_id"], "Jenis": row["device_type"], "EM Score": em,
                "DRI": round(dri, 3) if dri is not None else "Belum lengkap",
                "Prioritas": priority, "Alasan": " ".join(reasons),
                "Rekomendasi": " ".join(actions), "Usulan tindak lanjut": due_date,
                "Data DRI": "Lengkap" if dri is not None else "Belum lengkap"
            })
        risk_df = pd.DataFrame(risk_rows)
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
        selected = st.selectbox("Lihat penjelasan satu alat", assets["asset_id"].tolist(), key="risk_asset")
        chosen = risk_df[risk_df["ID Alat"] == selected].iloc[0]
        a, b = st.columns(2)
        with a:
            st.metric("EM Score", chosen["EM Score"])
            st.metric("DRI", chosen["DRI"])
            st.write("**Prioritas:**", chosen["Prioritas"])
        with b:
            st.write("**Alasan aturan:**", chosen["Alasan"])
            st.write("**Rekomendasi:**", chosen["Rekomendasi"])
            st.write("**Tanggal tindak lanjut yang diusulkan:**", chosen["Usulan tindak lanjut"])
        st.caption("Ambang DRI 0,50 dan 0,75 hanyalah ambang demonstrasi yang perlu ditetapkan/validasi oleh penelitian. DRI bukan probabilitas kegagalan.")
        if st.button("Buat work order untuk alat terpilih", type="primary"):
            with db_conn() as conn:
                conn.execute("""
                    INSERT INTO work_orders(asset_id, created_at, priority, reason, recommendation, due_date, status)
                    VALUES(?,?,?,?,?,?, 'Open')
                """, (selected, datetime.now().isoformat(timespec="seconds"), chosen["Prioritas"],
                      chosen["Alasan"], chosen["Rekomendasi"], chosen["Usulan tindak lanjut"]))
            audit(selected, "CREATE_WORK_ORDER", f"Prioritas {chosen['Prioritas']}; {chosen['Alasan']}")
            st.success("Work order dibuat.")
            st.rerun()

with tab_test:
    st.subheader("Pencatatan hasil uji / kalibrasi")
    assets = load_assets()
    if assets.empty:
        st.info("Daftarkan alat terlebih dahulu.")
    else:
        selected_id = st.selectbox("ID alat", assets["asset_id"].tolist(), key="test_asset")
        asset_row = assets[assets["asset_id"] == selected_id].iloc[0]
        demo = DEMO_LIMITS.get(asset_row["device_type"], DEMO_LIMITS["Other"])
        st.info(f"Parameter contoh: {demo['parameter']} | Unit: {demo['unit']} | Batas konfigurasi demo: ±{demo['limit_pct']}%. Verifikasi sebelum dipakai.")
        with st.form("test_form"):
            test_date = st.date_input("Tanggal uji", value=date.today())
            parameter_name = st.text_input("Nama parameter yang benar-benar diuji", value=demo["parameter"])
            unit = st.text_input("Satuan", value=demo["unit"])
            limit_pct = st.number_input("Batas error relatif (%) dari SOP tervalidasi", min_value=0.0, value=float(demo["limit_pct"]), step=0.1)
            ref = st.number_input("Nilai referensi/standar", value=10.0, step=0.1, format="%.4f")
            st.caption("Masukkan enam pembacaan. Gunakan satuan yang sama dengan nilai referensi.")
            cols = st.columns(3)
            readings = []
            for i in range(6):
                with cols[i % 3]:
                    readings.append(st.number_input(f"Pembacaan {i+1}", value=10.0, step=0.1, format="%.4f", key=f"reading_{i}"))
            notes = st.text_area("Catatan teknisi / nomor SOP / alat standar")
            submitted = st.form_submit_button("Hitung dan simpan hasil", type="primary", use_container_width=True)
        if submitted:
            if not parameter_name.strip() or not unit.strip():
                st.error("Nama parameter dan satuan wajib diisi.")
            elif ref == 0:
                st.error("Nilai referensi tidak boleh nol untuk perhitungan error relatif.")
            elif limit_pct <= 0:
                st.error("Batas error harus lebih besar dari nol dan berasal dari SOP yang tervalidasi.")
            else:
                mean_val = float(np.mean(readings))
                error = float(((mean_val - ref) / abs(ref)) * 100.0)
                result = "PASS" if abs(error) <= limit_pct else "FAIL"
                with db_conn() as conn:
                    conn.execute("""
                    INSERT INTO calibration_tests(asset_id, test_date, parameter_name, unit, reference_value,
                        readings_json, mean_value, error_pct, configured_limit_pct, result, notes)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """, (selected_id, test_date.isoformat(), parameter_name.strip(), unit.strip(), float(ref),
                          ",".join(str(x) for x in readings), mean_val, error, float(limit_pct), result, notes))
                    conn.execute("UPDATE assets SET deviation_pct=?, deviation_limit_pct=?, status=? WHERE asset_id=?",
                                 (abs(error), float(limit_pct), "Hasil uji perlu ditinjau" if result == "FAIL" else "Hasil uji PASS; tinjau sesuai prosedur", selected_id))
                audit(selected_id, "SAVE_TEST", f"{parameter_name}; mean={mean_val}; error={error:.4f}%; result={result}; limit={limit_pct}%")
                st.success("Hasil tersimpan.")
                st.metric("Rata-rata pembacaan", f"{mean_val:.4f} {unit}")
                st.metric("Error relatif", f"{error:.3f}%")
                if result == "PASS":
                    st.success("PASS menurut batas yang dimasukkan. Ini bukan sertifikat kalibrasi atau keputusan otomatis laik pakai.")
                else:
                    st.error("FAIL menurut batas yang dimasukkan. Minta peninjauan petugas berwenang dan ikuti SOP keselamatan.")
                st.rerun()
        history = load_tests(selected_id)
        if not history.empty:
            st.markdown("**Riwayat hasil uji alat ini**")
            st.dataframe(history.drop(columns=["readings_json"]), use_container_width=True, hide_index=True)

with tab_wo:
    st.subheader("Work Order")
    wo = load_work_orders()
    if wo.empty:
        st.info("Belum ada work order. Buat dari tab Risiko & Rekomendasi.")
    else:
        st.dataframe(wo, use_container_width=True, hide_index=True)
        wo_id = st.selectbox("Pilih nomor work order", wo["id"].astype(int).tolist())
        new_status = st.selectbox("Status baru", ["Open", "In Progress", "Completed", "Cancelled"])
        if st.button("Perbarui status work order"):
            with db_conn() as conn:
                conn.execute("UPDATE work_orders SET status=? WHERE id=?", (new_status, int(wo_id)))
            row = wo[wo["id"] == wo_id].iloc[0]
            audit(row["asset_id"], "UPDATE_WORK_ORDER", f"WO {wo_id} => {new_status}")
            st.success("Status work order diperbarui.")
            st.rerun()

with tab_history:
    st.subheader("Riwayat pengujian dan audit")
    tests = load_tests()
    if tests.empty:
        st.info("Belum ada hasil uji.")
    else:
        st.dataframe(tests.drop(columns=["readings_json"]), use_container_width=True, hide_index=True)
        st.download_button("Unduh riwayat uji CSV", tests.drop(columns=["readings_json"]).to_csv(index=False).encode("utf-8"), "medirisk_riwayat_uji.csv", "text/csv")
    with db_conn() as conn:
        audit_df = pd.read_sql_query("SELECT * FROM audit_log ORDER BY id DESC", conn)
    st.markdown("**Audit log**")
    if audit_df.empty:
        st.info("Belum ada aktivitas yang tercatat.")
    else:
        st.dataframe(audit_df, use_container_width=True, hide_index=True)
        st.download_button("Unduh audit log CSV", audit_df.to_csv(index=False).encode("utf-8"), "medirisk_audit_log.csv", "text/csv")

st.divider()
st.caption("MEDIRISK prototype | Gunakan data simulasi untuk demo. Jangan masukkan data pasien atau informasi sensitif.")
