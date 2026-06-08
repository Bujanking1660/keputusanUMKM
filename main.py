import os
import pandas as pd
import numpy as np

def calculate_knn_step_by_step(train_path, test_data, features, target_col, k_values=[3, 5]):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    
    # Load training data
    train_df = pd.read_csv(train_path)
    
    results = {}
    
    for q in test_data:
        q_name = q["Nama_UKM"]
        q_val = np.array([q[f] for f in features])
        
        # Copy to perform distance calculations
        df = train_df.copy()
        
        # Calculate squared distance
        sq_dists = []
        for _, row in df.iterrows():
            x_val = row[features].values.astype(float)
            sq_d = np.sum((x_val - q_val) ** 2)
            sq_dists.append(sq_d)
            
        df['Square_Distance'] = sq_dists
        df['Original_Index'] = df.index
        
        # Sort by square distance (with original index as secondary key for stable rank)
        sorted_df = df.sort_values(by=['Square_Distance', 'Original_Index']).reset_index(drop=True)
        sorted_df['Jarak_Terkecil_Rank'] = sorted_df.index + 1
        
        # Restore original order of the training dataset
        df = sorted_df.sort_values(by='Original_Index').reset_index(drop=True)
        
        # Construct the formula string
        formulas = []
        for _, row in df.iterrows():
            parts = []
            for f_name, q_v in zip(features, q_val):
                parts.append(f"({int(row[f_name])}-{int(q_v)})^2")
            expr = " + ".join(parts)
            formulas.append(f"{expr} = {int(row['Square_Distance'])}")
            
        df['Formula_Expr'] = formulas
        
        # Determine decisions
        decisions = {}
        for k in k_values:
            # For decisions, we take the top k closest neighbors
            nn_df = sorted_df.head(k)
            votes = nn_df[target_col].value_counts().to_dict()
            
            # Find the maximum vote count
            max_votes = max(votes.values())
            candidates = [label for label, count in votes.items() if count == max_votes]
            
            if len(candidates) > 1:
                # Tie-breaker: choose candidate with the closest single neighbor
                resolved = None
                for _, row_nn in nn_df.iterrows():
                    if row_nn[target_col] in candidates:
                        resolved = row_nn[target_col]
                        break
                decisions[k] = (resolved, votes, "TIE RESOLVED BY CLOSEST NEIGHBOR")
            else:
                decisions[k] = (candidates[0], votes, "UNANIMOUS/MAJORITY")
                
        results[q_name] = {
            "query": q,
            "df": df,
            "decisions": decisions
        }
        
    return results

def format_custom_markdown_table(df, q, k, features):
    q_val_str = ",".join(str(q[f]) for f in features)
    
    # Construct the columns requested by the user
    table_df = pd.DataFrame()
    table_df['Nama UKM'] = df['Nama_UKM']
    table_df['X1 = Lama Usaha (tahun)'] = df['Lama_Usaha']
    table_df['X2 = Jumlah Pekerja'] = df['Jumlah_Pekerja']
    table_df['X3 = Omzet (juta)'] = df['Omzet']
    table_df['X4 = Jumlah Aset'] = df['Jumlah_Aset']
    table_df[f'Square distance to query distance ({q_val_str})'] = df['Formula_Expr']
    table_df['Jarak Terkecil'] = df['Jarak_Terkecil_Rank']
    
    # Whether it is a nearest neighbor
    table_df['Apakah termasuk nearest neighbor (K)'] = df['Jarak_Terkecil_Rank'].apply(
        lambda r: 'Ya' if r <= k else 'Tidak'
    )
    
    # Y = label nearest neighbor (only shown if Ya, otherwise '-')
    table_df['Y = kategori nearest neighbor'] = df.apply(
        lambda r: r['Hasil_Keputusan'] if r['Jarak_Terkecil_Rank'] <= k else '-', axis=1
    )
    
    # Custom markdown formatting
    headers = " | ".join(str(c) for c in table_df.columns)
    headers = "| " + headers + " |"
    separator = " | ".join("---" for _ in table_df.columns)
    separator = "| " + separator + " |"
    lines = [headers, separator]
    for _, row in table_df.iterrows():
        row_str = " | ".join(str(val) for val in row.values)
        lines.append("| " + row_str + " |")
        
    return "\n".join(lines)

def generate_html_dashboard_custom(results, k_values=[3, 5]):
    # Pre-calculated decisions for Javascript integration
    js_decisions = {k: {} for k in k_values}
    for q_name, data in results.items():
        q_id = q_name.replace(" ", "-")
        for k in k_values:
            js_decisions[k][q_id] = data["decisions"][k][0]

    html_content = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>kNN UMKM Decision Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-light: #f8fafc;
            --card-light: #ffffff;
            --border-light: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --accent-green-solid: #10b981;
            
            /* Secondary Accent: Fresh light green gradient */
            --gradient-green: linear-gradient(135deg, #34d399, #10b981);
            --gradient-green-hover: linear-gradient(135deg, #10b981, #059669);
            --green-light: rgba(16, 185, 129, 0.08);
            --green-border: rgba(16, 185, 129, 0.2);
            --green-text: #047857;
            
            --accent-red: #ef4444;
            --accent-orange: #f59e0b;
            --red-light: #fee2e2;
            --orange-light: #fef3c7;
            --blue-light: #dbeafe;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        body {
            background-color: var(--bg-light);
            color: var(--text-main);
            padding: 2rem;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }
        
        h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.3rem;
            font-weight: 800;
            background: var(--gradient-green);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        }
        
        .subtitle {
            color: var(--text-muted);
            font-size: 1.05rem;
        }
        
        .query-tabs {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .tab-btn {
            background: var(--card-light);
            color: var(--text-muted);
            border: 1px solid var(--border-light);
            padding: 0.75rem 1.5rem;
            border-radius: 9999px;
            cursor: pointer;
            font-size: 0.95rem;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        
        .tab-btn:hover {
            color: var(--green-text);
            border-color: var(--accent-green-solid);
            transform: translateY(-2px);
        }
        
        .tab-btn.active {
            background: var(--gradient-green);
            color: #ffffff;
            border-color: transparent;
            box-shadow: 0 8px 12px -3px rgba(16, 185, 129, 0.3);
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 2rem;
        }
        
        @media (max-width: 1000px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: var(--card-light);
            border: 1px solid var(--border-light);
            border-radius: 1.25rem;
            padding: 1.75rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.03), 0 4px 6px -4px rgba(0, 0, 0, 0.03);
            margin-bottom: 1.5rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 20px -3px rgba(0, 0, 0, 0.06);
        }
        
        .query-info h3 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
            border-bottom: 2px solid var(--border-light);
            padding-bottom: 0.5rem;
        }
        
        .feature-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .feature-badge {
            background: var(--bg-light);
            border: 1px solid var(--border-light);
            padding: 0.75rem;
            border-radius: 0.75rem;
            text-align: center;
        }
        
        .feature-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }
        
        .feature-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-main);
        }
        
        .decision-box {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.04), rgba(52, 211, 153, 0.04));
            border: 1px dashed var(--accent-green-solid);
            border-radius: 1rem;
            padding: 1.25rem;
            text-align: center;
            margin-top: 1rem;
        }
        
        .decision-title {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }
        
        .decision-val {
            font-size: 1.75rem;
            font-weight: 800;
            font-family: 'Outfit', sans-serif;
            letter-spacing: 0.05em;
        }
        
        .decision-val.YA {
            color: var(--accent-green-solid);
        }
        
        .decision-val.TIDAK {
            color: var(--accent-red);
        }
        
        .decision-val.TUNDA {
            color: var(--accent-orange);
        }
        
        .table-container {
            overflow-x: auto;
            margin-top: 0.5rem;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.85rem;
        }
        
        th {
            background-color: #f1f5f9;
            color: var(--text-muted);
            font-weight: 600;
            padding: 0.75rem;
            border-bottom: 2px solid var(--border-light);
            font-size: 0.75rem;
            text-transform: uppercase;
        }
        
        td {
            padding: 0.75rem;
            border-bottom: 1px solid var(--border-light);
            color: var(--text-main);
        }
        
        tr.active-query-row {
            background-color: var(--green-light);
            border-left: 4px solid var(--accent-green-solid);
        }
        
        tr.nn-row {
            background-color: rgba(52, 211, 153, 0.08);
        }
        
        .badge {
            padding: 0.25rem 0.5rem;
            border-radius: 0.375rem;
            font-weight: 700;
            font-size: 0.75rem;
        }
        
        .badge.YA {
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--green-text);
        }
        
        .badge.TIDAK {
            background-color: var(--red-light);
            color: var(--accent-red);
        }
        
        .badge.TUNDA {
            background-color: var(--orange-light);
            color: var(--accent-orange);
        }
        
        .nn-badge {
            font-weight: 700;
            padding: 0.2rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
        }
        
        .nn-badge.yes {
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--green-text);
        }
        
        .nn-badge.no {
            color: var(--text-muted);
            opacity: 0.6;
        }
        
        .content-panel {
            display: none;
        }
        
        .content-panel.active {
            display: block;
        }
        
        .k-toggle {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 1.5rem;
            gap: 0.5rem;
            align-items: center;
        }
        
        .k-btn {
            background: var(--card-light);
            color: var(--text-muted);
            border: 1px solid var(--border-light);
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        
        .k-btn.active {
            background: var(--gradient-green);
            color: #ffffff;
            border-color: transparent;
            box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
        }
        
        .formula-col {
            font-family: monospace;
            font-size: 0.8rem;
            color: #475569;
        }
        
        .section-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .section-title::before {
            content: '';
            display: inline-block;
            width: 4px;
            height: 18px;
            background: var(--gradient-green);
            border-radius: 2px;
        }

        /* Export PNG Button */
        .export-btn {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: #fff;
            border: none;
            padding: 0.5rem 1.1rem;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 600;
            font-family: 'Plus Jakarta Sans', sans-serif;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
            letter-spacing: 0.01em;
            margin-left: auto;
        }
        .export-btn:hover {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            box-shadow: 0 6px 18px rgba(99, 102, 241, 0.45);
            transform: translateY(-2px);
        }
        .export-btn:active {
            transform: translateY(0);
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
        }
        .export-btn svg { flex-shrink: 0; }
        .export-btn.loading {
            opacity: 0.7;
            cursor: not-allowed;
            pointer-events: none;
        }
        /* Toast notification */
        #export-toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: linear-gradient(135deg, #1e1b4b, #312e81);
            color: #fff;
            padding: 0.85rem 1.4rem;
            border-radius: 0.75rem;
            font-size: 0.875rem;
            font-weight: 600;
            box-shadow: 0 10px 25px rgba(99, 102, 241, 0.35);
            z-index: 9999;
            opacity: 0;
            transform: translateY(1rem);
            transition: opacity 0.3s ease, transform 0.3s ease;
            pointer-events: none;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        #export-toast.show { opacity: 1; transform: translateY(0); }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>Visualisasi Algoritma kNN - Keputusan Bantuan UMK</h1>
            <p class="subtitle">Simulasi perhitungan jarak Euclidean (D²) berdasarkan kriteria kelayakan modal</p>
        </header>
        
        <div class="query-tabs">
"""
    
    for idx, q_name in enumerate(results.keys()):
        active_class = "active" if idx == 0 else ""
        html_content += f'            <button class="tab-btn {active_class}" onclick="switchTab(event, \'{q_name.replace(" ", "-")}\')">{q_name}</button>\n'
        
    html_content += """        </div>
        
        <div class="main-content">
"""
    
    for idx, (q_name, data) in enumerate(results.items()):
        active_class = "active" if idx == 0 else ""
        q = data["query"]
        df = data["df"]
        decisions = data["decisions"]
        q_val_str = f"{q['Lama_Usaha']},{q['Jumlah_Pekerja']},{q['Omzet']},{q['Jumlah_Aset']}"
        q_id = q_name.replace(" ", "-")
        
        html_content += f'            <div id="{q_id}" class="content-panel {active_class}">\n'
        html_content += f'                <div class="k-toggle">\n'
        html_content += f'                    <span style="font-size: 0.85rem; color: var(--text-muted); font-weight: 500;">Pilih Parameter k:</span>\n'
        html_content += f'                    <button class="k-btn active" onclick="setK(3)">k = 3</button>\n'
        html_content += f'                    <button class="k-btn" onclick="setK(5)">k = 5</button>\n'
        html_content += f'                    <button class="export-btn" id="export-btn-{q_id}" onclick="exportToPNG(\'{q_id}\')">\n'
        html_content += f'                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>\n'
        html_content += f'                        Export PNG\n'
        html_content += f'                    </button>\n'
        html_content += f'                </div>\n'
        html_content += """                
                <div class="dashboard-grid">
                    <!-- Left Panel: Profil & Decision Box -->
                    <div class="card query-info" style="align-self: start;">
                        <h3>Profil UMK Uji (Query)</h3>
                        <div class="feature-grid">
                            <div class="feature-badge">
                                <div class="feature-label">Lama Usaha</div>
                                <div class="feature-value">""" + f'{q["Lama_Usaha"]} th' + """</div>
                            </div>
                            <div class="feature-badge">
                                <div class="feature-label">Pekerja</div>
                                <div class="feature-value">""" + f'{q["Jumlah_Pekerja"]}' + """</div>
                            </div>
                            <div class="feature-badge">
                                <div class="feature-label">Omzet</div>
                                <div class="feature-value">""" + f'{q["Omzet"]} jt' + """</div>
                            </div>
                            <div class="feature-badge">
                                <div class="feature-label">Aset</div>
                                <div class="feature-value">""" + f'{q["Jumlah_Aset"]}' + """</div>
                            </div>
                        </div>
                        
                        <div class="decision-box-container">
                            <div id="dec-box-3-""" + q_id + """" class="decision-box">
                                <div class="decision-title">Hasil Keputusan (k=3)</div>
                                <div class="decision-val """ + f'{decisions[3][0]}' + """">""" + f'{decisions[3][0]}' + """</div>
                                <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500;">
                                    Voting: """ + f'{decisions[3][1]}' + """
                                </div>
                            </div>
                            
                            <div id="dec-box-5-""" + q_id + """" class="decision-box" style="display: none; border-color: var(--accent-green-solid);">
                                <div class="decision-title">Hasil Keputusan (k=5)</div>
                                <div class="decision-val """ + f'{decisions[5][0]}' + """">""" + f'{decisions[5][0]}' + """</div>
                                <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500;">
                                    Voting: """ + f'{decisions[5][1]}' + """
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Right Panel: Stacked Tables -->
                    <div style="display: flex; flex-direction: column;">
                        
                        <!-- CARD 1: Tabel UMK Uji & Keputusan Akhir (Summary Table) -->
                        <div class="card">
                            <div class="section-title">Tabel 2: Data Uji & Hasil Keputusan Akhir (kNN)</div>
                            <div class="table-container">
                                <table class="test-summary-table">
                                    <thead>
                                        <tr>
                                            <th>NAMA UMK</th>
                                            <th>LAMA USAHA (dlm tahun)</th>
                                            <th>JUMLAH PEKERJA</th>
                                            <th>OMZET (dlm juta)</th>
                                            <th>JUMLAH ASET</th>
                                            <th>HASIL KEPUTUSAN</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr id="row-t2-Umk-Pamurbaya" class="t2-row">
                                            <td style="font-weight: 700;">Umk Pamurbaya</td>
                                            <td>4</td>
                                            <td>15</td>
                                            <td>4</td>
                                            <td>6</td>
                                            <td id="t2-dec-Umk-Pamurbaya" style="font-weight: 700;">
                                                <span class="badge TUNDA">TUNDA</span>
                                            </td>
                                        </tr>
                                        <tr id="row-t2-Umk-Bandeng" class="t2-row">
                                            <td style="font-weight: 700;">Umk Bandeng</td>
                                            <td>3</td>
                                            <td>28</td>
                                            <td>4</td>
                                            <td>10</td>
                                            <td id="t2-dec-Umk-Bandeng" style="font-weight: 700;">
                                                <span class="badge TUNDA">TUNDA</span>
                                            </td>
                                        </tr>
                                        <tr id="row-t2-Bukid-Graffer" class="t2-row">
                                            <td style="font-weight: 700;">Bukid Graffer</td>
                                            <td>2</td>
                                            <td>12</td>
                                            <td>1</td>
                                            <td>3</td>
                                            <td id="t2-dec-Bukid-Graffer" style="font-weight: 700;">
                                                <span class="badge TUNDA">TUNDA</span>
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        
                        <!-- CARD 2: Tabel Simulasi Rumus & Jarak (Calculation Table) -->
                        <div class="card">
                            <div class="section-title">Tabel 1: Simulasi Lembar Perhitungan Jarak (kNN)</div>
                            <div class="table-container">
                                <table class="calculation-table" id="calc-table-""" + q_id + """">
                                    <thead>
                                        <tr>
                                            <th>Nama UKM Latih</th>
                                            <th>X1 = Lama Usaha (tahun)</th>
                                            <th>X2 = Jml Pekerja</th>
                                            <th>X3 = Omzet (juta)</th>
                                            <th>X4 = Jml Aset</th>
                                            <th>Square distance to query distance (""" + q_val_str + """)</th>
                                            <th>Jarak Terkecil</th>
                                            <th>Apakah termasuk nearest neighbor (K)</th>
                                            <th>Y = kategori nearest neighbor</th>
                                        </tr>
                                    </thead>
                                    <tbody>
"""
        
        for _, row in df.iterrows():
            rank = int(row['Jarak_Terkecil_Rank'])
            
            html_content += f'                                        <tr data-rank="{rank}">\n'
            html_content += f'                                            <td style="font-weight: 700;">{row["Nama_UKM"]}</td>\n'
            html_content += f'                                            <td>{row["Lama_Usaha"]}</td>\n'
            html_content += f'                                            <td>{row["Jumlah_Pekerja"]}</td>\n'
            html_content += f'                                            <td>{row["Omzet"]}</td>\n'
            html_content += f'                                            <td>{row["Jumlah_Aset"]}</td>\n'
            html_content += f'                                            <td class="formula-col">{row["Formula_Expr"]}</td>\n'
            html_content += f'                                            <td style="font-weight: 700; text-align: center;">{rank}</td>\n'
            
            # Default to k=3 highlights
            is_nn = "Ya" if rank <= 3 else "Tidak"
            nn_badge_class = "yes" if rank <= 3 else "no"
            label = row[target_col] if rank <= 3 else "-"
            row_highlight = "class=\"nn-row\"" if rank <= 3 else ""
            
            html_content += f'                                            <td class="nn-status-cell" {row_highlight}><span class="nn-badge {nn_badge_class}">{is_nn}</span></td>\n'
            
            label_badge = f'<span class="badge {label}">{label}</span>' if label != "-" else "-"
            html_content += f'                                            <td class="nn-label-cell" style="text-align: center;">{label_badge}</td>\n'
            html_content += f'                                        </tr>\n'
            
        html_content += """                                    </tbody>
                                </table>
                            </div>
                        </div>
                        
                    </div>
                </div>
            </div>
"""
        
    html_content += """        </div>
    </div>
    
    <script>
        // Global precalculated decisions from Python
        const decisions = """ + str(js_decisions) + """;
        
        // Initial state
        let currentK = 3;
        let currentQueryId = "Umk-Pamurbaya";
        
        window.onload = function() {
            updateRowHighlights();
        };

        function switchTab(evt, qId) {
            currentQueryId = qId;
            
            // Hide all panels
            var tabcontent = document.getElementsByClassName("content-panel");
            for (var i = 0; i < tabcontent.length; i++) {
                tabcontent[i].classList.remove("active");
            }

            // Remove active tabs
            var tablinks = document.getElementsByClassName("tab-btn");
            for (var i = 0; i < tablinks.length; i++) {
                tablinks[i].classList.remove("active");
            }

            // Active current
            document.getElementById(qId).classList.add("active");
            evt.currentTarget.classList.add("active");
            
            // Re-apply k configuration to the newly shown panel
            setK(currentK);
            updateRowHighlights();
        }
        
        function setK(kVal) {
            currentK = kVal;
            
            // Update active state of buttons in all panels
            var btnContainers = document.getElementsByClassName("k-toggle");
            for (var c = 0; c < btnContainers.length; c++) {
                var buttons = btnContainers[c].getElementsByClassName("k-btn");
                for (var i = 0; i < buttons.length; i++) {
                    buttons[i].classList.remove("active");
                    if (buttons[i].innerText === "k = " + kVal) {
                        buttons[i].classList.add("active");
                    }
                }
            }
            
            // Update decision boxes for all query panels
            var tabcontent = document.getElementsByClassName("content-panel");
            for (var i = 0; i < tabcontent.length; i++) {
                var qId = tabcontent[i].getAttribute("id");
                if (kVal === 3) {
                    document.getElementById("dec-box-3-" + qId).style.display = "block";
                    document.getElementById("dec-box-5-" + qId).style.display = "none";
                } else {
                    document.getElementById("dec-box-3-" + qId).style.display = "none";
                    document.getElementById("dec-box-5-" + qId).style.display = "block";
                }
            }
            
            // Update Table 2 (Summary Decisions)
            document.getElementById("t2-dec-Umk-Pamurbaya").innerHTML = '<span class="badge ' + decisions[kVal]["Umk-Pamurbaya"] + '">' + decisions[kVal]["Umk-Pamurbaya"] + '</span>';
            document.getElementById("t2-dec-Umk-Bandeng").innerHTML = '<span class="badge ' + decisions[kVal]["Umk-Bandeng"] + '">' + decisions[kVal]["Umk-Bandeng"] + '</span>';
            document.getElementById("t2-dec-Bukid-Graffer").innerHTML = '<span class="badge ' + decisions[kVal]["Bukid-Graffer"] + '">' + decisions[kVal]["Bukid-Graffer"] + '</span>';
            
            // Update Table 1 (Calculation Steps for all tabs)
            for (var t = 0; t < tabcontent.length; t++) {
                var qId = tabcontent[t].getAttribute("id");
                var calcTable = document.getElementById("calc-table-" + qId);
                var rows = calcTable.getElementsByTagName("tbody")[0].getElementsByTagName("tr");
                
                for (var i = 0; i < rows.length; i++) {
                    var row = rows[i];
                    var rank = parseInt(row.getAttribute("data-rank"));
                    
                    var nnCell = row.getElementsByClassName("nn-status-cell")[0];
                    var labelCell = row.getElementsByClassName("nn-label-cell")[0];
                    
                    if (rank <= kVal) {
                        row.classList.add("nn-row");
                        nnCell.innerHTML = '<span class="nn-badge yes">Ya</span>';
                        
                        var originalLabel = "";
                        if (qId.includes("Pamurbaya")) {
                            originalLabel = getPamurbayaLabel(rank);
                        } else if (qId.includes("Bandeng")) {
                            originalLabel = getBandengLabel(rank);
                        } else {
                            originalLabel = getBukidLabel(rank);
                        }
                        labelCell.innerHTML = '<span class="badge ' + originalLabel + '">' + originalLabel + '</span>';
                    } else {
                        row.classList.remove("nn-row");
                        nnCell.innerHTML = '<span class="nn-badge no">Tidak</span>';
                        labelCell.innerHTML = '-';
                    }
                }
            }
        }
        
        function updateRowHighlights() {
            // In Table 2, highlight only the row corresponding to the active tab query
            var summaryRows = document.getElementsByClassName("t2-row");
            for (var i = 0; i < summaryRows.length; i++) {
                summaryRows[i].classList.remove("active-query-row");
            }
            var activeRow = document.getElementById("row-t2-" + currentQueryId);
            if (activeRow) {
                activeRow.classList.add("active-query-row");
            }
        }
        
        function getPamurbayaLabel(rank) {
            var labels = {
                1: "TUNDA", 2: "TIDAK", 3: "YA", 4: "TIDAK", 5: "TUNDA",
                6: "TUNDA", 7: "TIDAK", 8: "TUNDA", 9: "TUNDA", 10: "TIDAK",
                11: "TUNDA", 12: "TUNDA", 13: "TIDAK", 14: "YA", 15: "YA", 16: "YA"
            };
            return labels[rank] || "-";
        }
        
        function getBandengLabel(rank) {
            var labels = {
                1: "TIDAK", 2: "TUNDA", 3: "TUNDA", 4: "TUNDA", 5: "TUNDA",
                6: "YA", 7: "TIDAK", 8: "TIDAK", 9: "TIDAK", 10: "TUNDA",
                11: "TUNDA", 12: "TUNDA", 13: "TIDAK", 14: "YA", 15: "YA", 16: "YA"
            };
            return labels[rank] || "-";
        }
        
        function getBukidLabel(rank) {
            var labels = {
                1: "TUNDA", 2: "TIDAK", 3: "TUNDA", 4: "TIDAK", 5: "TUNDA",
                6: "TIDAK", 7: "YA", 8: "TUNDA", 9: "TUNDA", 10: "TIDAK",
                11: "YA", 12: "TUNDA", 13: "YA", 14: "YA", 15: "TUNDA", 16: "TIDAK"
            };
            return labels[rank] || "-";
        }

        // \u2500\u2500 Export to PNG \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        function showToast(msg) {
            var toast = document.getElementById('export-toast');
            toast.querySelector('.toast-msg').textContent = msg;
            toast.classList.add('show');
            setTimeout(function() { toast.classList.remove('show'); }, 3000);
        }

        function exportToPNG(qId) {
            var panel = document.getElementById(qId);
            if (!panel) return;
            var btn = document.getElementById('export-btn-' + qId);
            if (btn) {
                btn.classList.add('loading');
                btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> Memproses...';
            }
            var nameMap = {};
            var tabcontent = document.getElementsByClassName('content-panel');
            for (var i = 0; i < tabcontent.length; i++) {
                var id = tabcontent[i].getAttribute('id');
                nameMap[id] = id.replace(/-/g, '');
            }
            var safeName = nameMap[qId] || qId.replace(/-/g, '');
            var filename = 'kNN_' + safeName + '_k' + currentK + '.png';
            var originalDisplay = panel.style.display;
            panel.style.display = 'block';
            var tableContainers = panel.querySelectorAll('.table-container');
            var savedOverflows = [];
            tableContainers.forEach(function(tc, i) {
                savedOverflows[i] = tc.style.overflowX;
                tc.style.overflowX = 'visible';
            });
            html2canvas(panel, {
                scale: 2,
                useCORS: true,
                backgroundColor: '#f8fafc',
                logging: false,
                scrollX: 0,
                scrollY: 0,
                windowWidth: panel.scrollWidth + 80,
                width: panel.scrollWidth + 80
            }).then(function(canvas) {
                tableContainers.forEach(function(tc, i) { tc.style.overflowX = savedOverflows[i]; });
                panel.style.display = originalDisplay;
                var link = document.createElement('a');
                link.download = filename;
                link.href = canvas.toDataURL('image/png');
                link.click();
                if (btn) {
                    btn.classList.remove('loading');
                    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Export PNG';
                }
                showToast('\\u2705 ' + filename + ' berhasil diunduh!');
            }).catch(function(err) {
                console.error('Export PNG error:', err);
                tableContainers.forEach(function(tc, i) { tc.style.overflowX = savedOverflows[i]; });
                panel.style.display = originalDisplay;
                if (btn) {
                    btn.classList.remove('loading');
                    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Export PNG';
                }
                showToast('\\u274C Gagal mengekspor. Coba lagi.');
            });
        }
    </script>

    <!-- Toast Notification -->
    <div id="export-toast">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        <span class="toast-msg">Mengunduh...</span>
    </div>
</body>
</html>
"""
    return html_content

if __name__ == "__main__":
    train_path = 'data/data.csv'
    features = ["Lama_Usaha", "Jumlah_Pekerja", "Omzet", "Jumlah_Aset"]
    target_col = "Hasil_Keputusan"
    
    test_data = [
        {"Nama_UKM": "Umk Pamurbaya", "Lama_Usaha": 4, "Jumlah_Pekerja": 15, "Omzet": 4, "Jumlah_Aset": 6},
        {"Nama_UKM": "Umk Bandeng", "Lama_Usaha": 3, "Jumlah_Pekerja": 28, "Omzet": 4, "Jumlah_Aset": 10},
        {"Nama_UKM": "Bukid Graffer", "Lama_Usaha": 2, "Jumlah_Pekerja": 12, "Omzet": 1, "Jumlah_Aset": 3}
    ]
    
    # Run
    results = calculate_knn_step_by_step(train_path, test_data, features, target_col)
    
    # Print to console for each query (using k=3 as standard)
    for q_name, data in results.items():
        q = data["query"]
        df = data["df"]
        decisions = data["decisions"]
        
        print("="*100)
        print(f"ANALISIS kNN UNTUK: {q_name}")
        print(f"Data Uji: Lama Usaha={q['Lama_Usaha']}, Pekerja={q['Jumlah_Pekerja']}, Omzet={q['Omzet']}, Aset={q['Jumlah_Aset']}")
        print(f"Keputusan (k=3): {decisions[3][0]} (Votes: {decisions[3][1]} - {decisions[3][2]})")
        print(f"Keputusan (k=5): {decisions[5][0]} (Votes: {decisions[5][1]} - {decisions[5][2]})")
        print("-"*100)
        print("TABEL SIMULASI RUMUS & JARAK (k=3):")
        print(format_custom_markdown_table(df, q, 3, features))
        print("\n")
        
    # Generate HTML dashboard
    html_dashboard = generate_html_dashboard_custom(results)
    with open('dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_dashboard)
    print("HTML dashboard updated with custom formula tables in 'dashboard.html'")
