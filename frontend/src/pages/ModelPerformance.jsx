import React, { useMemo } from 'react';
import { useAppContext, formatPercent, titleize } from '../App';
import { Sparkles, ArrowRightLeft, Info, AlertTriangle, ShieldCheck } from 'lucide-react';

export default function ModelPerformance() {
  const { metrics, features, ellipticMetrics } = useAppContext();

  // Exclude logistic regression as its performance is too low to be useful
  const modelEntries = Object.entries(metrics?.metrics || {}).filter(([name]) => name !== 'logistic_regression');
  const bestModelName = metrics?.best_model || 'random_forest';

  const maxImportance = features?.length
    ? Math.max(...features.map(f => f.importance))
    : 1;

  // Model display order: best model first, then GNN
  const sortedEntries = useMemo(() => {
    return [...modelEntries].sort(([a], [b]) => {
      if (a === bestModelName) return -1;
      if (b === bestModelName) return 1;
      return 0;
    });
  }, [modelEntries, bestModelName]);

  const renderMetricPills = (m, modelName) => {
    if (!m) return null;
    const isRF = modelName === 'random_forest';
    const pills = [
      { label: 'Precision', value: m.precision, desc: 'Fraction of flagged cases that were actually fraud' },
      { label: 'Recall', value: m.recall, desc: 'Fraction of actual fraud cases caught by the model' },
      { label: 'F1 Score', value: m.f1_score, desc: 'Harmonic mean of Precision and Recall' },
      { label: 'Avg Precision', value: m.average_precision, desc: 'Area under Precision-Recall Curve (PR-AUC)' },
      { label: 'ROC-AUC', value: m.roc_auc, desc: 'Area under Receiver Operating Characteristic curve' },
      {
        label: 'Test Support',
        rawValue: isRF ? '49,963 legit / 37 fraud' : '19,976 legit / 24 fraud',
        desc: 'Number of legitimate and fraud cases in the held-out evaluation test set'
      },
    ];
    return (
      <div className="metrics-row">
        {pills.map(p => (
          <div className="metric-pill" key={p.label} title={p.desc}>
            <span className="metric-pill-label">{p.label}</span>
            <span className="metric-pill-value">
              {p.rawValue ? p.rawValue : formatPercent(p.value)}
            </span>
          </div>
        ))}
      </div>
    );
  };

  const renderConfusionMatrix = (modelName, modelData, showExplanation = false) => {
    if (!modelData?.confusion_matrix) return null;
    const matrix = modelData.confusion_matrix;
    const TN = matrix[0][0];
    const FP = matrix[0][1];
    const FN = matrix[1][0];
    const TP = matrix[1][1];
    const isRF = modelName === 'random_forest';
    const totalTest = TN + FP + FN + TP;
    const totalFraud = FN + TP;

    return (
      <div className="confusion-section" key={modelName} style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px', marginBottom: '8px' }}>
          <h3 className="confusion-title" style={{ margin: 0 }}>
            Confusion Matrix — {titleize(modelName)} ({totalTest.toLocaleString()} Held-Out Test Transactions)
          </h3>
          <span className="risk-badge low" style={{ textTransform: 'none', fontSize: '0.72rem' }}>
            {totalFraud} Ground-Truth Fraud Cases in Test Split
          </span>
        </div>
        <div className="confusion-grid">
          <div className="confusion-cell tn">
            <span className="confusion-label">True Negative (Normal transactions correctly approved)</span>
            <span className="confusion-value">{TN.toLocaleString()}</span>
          </div>
          <div className="confusion-cell fp">
            <span className="confusion-label">False Positive (Normal transactions incorrectly flagged as fraud)</span>
            <span className="confusion-value">{FP.toLocaleString()}</span>
          </div>
          <div className="confusion-cell fn">
            <span className="confusion-label">False Negative (Fraudulent transactions missed)</span>
            <span className="confusion-value">{FN.toLocaleString()}</span>
          </div>
          <div className="confusion-cell tp">
            <span className="confusion-label">True Positive (Fraudulent transactions correctly caught)</span>
            <span className="confusion-value">{TP.toLocaleString()}</span>
          </div>
        </div>
        {showExplanation && (
          <div className="explanation-text" style={{ marginTop: '8px' }}>
            <p style={{ margin: '0 0 4px 0' }}>
              <strong>False Positive ({FP}):</strong> Legitimate customer payments flagged by mistake. On this 50,000-record test set, Random Forest produced 0 false positives.
            </p>
            <p style={{ margin: 0 }}>
              <strong>False Negative ({FN}):</strong> Actual fraud that escaped detection. Random Forest missed 1 fraud transaction out of 37, yielding 97.30% recall.
            </p>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="page-content page-enter">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px', marginBottom: '4px' }}>
        <h2 className="section-title" style={{ margin: 0 }}>Model Performance & Evaluation Audit</h2>
        <span className="risk-badge medium" style={{ textTransform: 'none', fontSize: '0.75rem' }}>
          Held-Out Test Sample Evaluations (FinGraph PaySim)
        </span>
      </div>
      <p className="section-subtitle" style={{ marginBottom: '18px' }}>
        Performance metrics evaluated strictly on held-out test splits. Metrics reflect the experimental sample and do not extrapolate uncritically to unseen time periods.
      </p>

      {/* ── Model Comparison Cards ── */}
      <div className="model-comparison" style={{
        gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
        marginBottom: '24px',
      }}>
        {sortedEntries.map(([name, m]) => {
          const isBest = name === bestModelName;
          const isRF = name === 'random_forest';
          return (
            <div className={`model-card${isBest ? ' best' : ''}`} key={name}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '4px' }}>
                <div>
                  <div className="model-name">{isRF ? 'Random Forest (Tabular Baseline)' : 'GraphSAGE (Heterogeneous GNN)'}</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    {isRF
                      ? 'Evaluated on 50,000 held-out test records (75/25 stratified split of 200k sample)'
                      : 'Evaluated on 20,000 held-out test records (80/20 random split of 100k sample)'}
                  </div>
                </div>
                <span className={`model-tag ${isBest ? 'best' : 'comparison'}`}>
                  {isBest ? 'Best Model' : 'GNN Model'}
                </span>
              </div>
              {renderMetricPills(m, name)}
              {!isRF && (
                <div style={{ fontSize: '0.72rem', color: 'var(--amber-text)', marginTop: '8px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <AlertTriangle size={12} />
                  <span>Note: Decision threshold was tuned directly on the test set during experiment search.</span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* ── Confusion Matrices ── */}
      {sortedEntries.map(([name, m]) => renderConfusionMatrix(name, m, name === bestModelName))}

      {/* ── Feature Importance ── */}
      <div className="dash-card" style={{ marginBottom: '24px' }}>
        <div className="card-header">
          <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sparkles size={18} />
            Random Forest Feature Importance
          </span>
          <span className="text-muted text-xs">Derived from scikit-learn Gini importance</span>
        </div>
        <p className="card-subtitle" style={{ marginBottom: '12px' }}>
          Engineered balance mismatch and account emptying indicators account for over 45% of total predictive power.
        </p>
        <div className="feature-chart">
          {(features || []).map((f, i) => {
            const pct = maxImportance > 0 ? (f.importance / maxImportance) * 100 : 0;
            return (
              <div className="feature-row" key={i}>
                <div className="feature-info">
                  <span className="feature-name">{f.feature}</span>
                  <span className="feature-pct">{formatPercent(f.importance)}</span>
                </div>
                <div className="feature-track">
                  <div className="feature-bar" style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Dataset Comparison: FinGraph vs Elliptic (Literature Baseline) ── */}
      <div className="dash-card" style={{ marginBottom: '24px' }}>
        <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <ArrowRightLeft size={18} />
              Cross-Dataset Analysis: Graph Topology Impact on GNN Performance
            </span>
            <p className="card-subtitle">
              Theoretical comparison of GNN effectiveness across dense cryptocurrency graphs versus sparse mobile money transactions.
            </p>
          </div>
          <span className="risk-badge medium" style={{ textTransform: 'none', fontSize: '0.72rem' }}>
            External Benchmark (Illustrative Only — Elliptic Dataset Absent)
          </span>
        </div>

        <div className="notice-banner warning" style={{ margin: '8px 0 16px 0', fontSize: '0.78rem' }}>
          <Info size={14} />
          <span>
            <strong>Disclaimer:</strong> The Elliptic Bitcoin dataset files are not bundled with this project repository. The Elliptic metrics below are hardcoded literature baseline reference values, not live FinGraph evaluation results.
          </span>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Property</th>
                <th>FinGraph PaySim (Evaluated on Local Sample)</th>
                <th>Elliptic Bitcoin (External Literature Reference)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ fontWeight: 700, color: 'var(--text)' }}>Transactions Analyzed</td>
                <td>{(metrics?.sample_rows || 200000).toLocaleString()} (50,000 test set)</td>
                <td>{ellipticMetrics ? ellipticMetrics.total_transactions?.toLocaleString() : '203,769'} (16,670 test set)</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 700, color: 'var(--text)' }}>Graph Connectivity</td>
                <td style={{ color: 'var(--amber-text)' }}>Sparse bipartite — most accounts transact once</td>
                <td style={{ color: 'var(--green-text)' }}>{ellipticMetrics ? ellipticMetrics.edges?.toLocaleString() : '234,355'} dense multi-hop transaction flows</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 700, color: 'var(--text)' }}>Fraud Class Rate</td>
                <td>{formatPercent(metrics?.fraud_rate_in_sample)} (0.074% in sample)</td>
                <td>~2.2% illicit transaction addresses</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 700, color: 'var(--text)' }}>Superior Model</td>
                <td>
                  <span className="risk-badge low">Random Forest (F1: 98.6%)</span>
                </td>
                <td>
                  <span className="risk-badge critical">GraphSAGE / GNN (F1: ~71–80%)</span>
                </td>
              </tr>
              <tr>
                <td style={{ fontWeight: 700, color: 'var(--text)' }}>Random Forest F1</td>
                <td style={{ fontWeight: 700 }}>
                  {formatPercent(metrics?.metrics?.random_forest?.f1_score)}
                </td>
                <td>
                  {ellipticMetrics
                    ? formatPercent(ellipticMetrics.metrics?.random_forest?.f1_score)
                    : '76.0%'}
                </td>
              </tr>
              <tr>
                <td style={{ fontWeight: 700, color: 'var(--text)' }}>GNN / GraphSAGE F1</td>
                <td>
                  {formatPercent(metrics?.metrics?.graph_sage?.f1_score)}
                </td>
                <td style={{ fontWeight: 700 }}>
                  {ellipticMetrics
                    ? formatPercent(ellipticMetrics.metrics?.graph_sage?.f1_score)
                    : '70.7%'}
                </td>
              </tr>
              <tr>
                <td style={{ fontWeight: 700, color: 'var(--text)' }}>Topological Rationale</td>
                <td style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
                  Disconnected star pairs limit neighbor message passing. Fraud indicators are captured cleanly by tabular balance error formulas.
                </td>
                <td style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
                  Multi-hop laundering chains allow graph convolution to aggregate structural indicators across wallet neighborhoods.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Academic Disclosures & Methodology Limitations ── */}
      <div className="dash-card" style={{ borderLeft: '4px solid #f59e0b' }}>
        <div className="card-header" style={{ marginBottom: '8px' }}>
          <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertTriangle size={18} className="text-warning" />
            Academic Presentation Disclosures & Limitations
          </span>
        </div>
        <div className="explanation-text" style={{ fontSize: '0.82rem', lineHeight: '1.6', color: 'var(--text-secondary)' }}>
          <p style={{ margin: '0 0 8px 0' }}>
            <strong>1. Early-Timestep Random Split:</strong> Both models were evaluated on early time-step samples (steps 1–9) using random splitting rather than a temporal out-of-time split (e.g. training on Days 1–20 and evaluating on Days 21–30). In operational environments, temporal drift and concept shift may affect generalization.
          </p>
          <p style={{ margin: '0 0 8px 0' }}>
            <strong>2. Limited Fraud Support in Test Split:</strong> Out of 50,000 held-out test transactions, only 37 are positive fraud cases (0.074%). A single prediction error shifts recall by 2.70% (1/37). High F1 scores must be interpreted within this narrow positive support.
          </p>
          <p style={{ margin: '0 0 8px 0' }}>
            <strong>3. GNN Threshold Calibration:</strong> GraphSAGE&apos;s decision threshold was selected by optimizing F1 directly against test-set scores across 8 random seeds. In production benchmarking, hyperparameter and threshold selection must be isolated to a separate validation split.
          </p>
          <p style={{ margin: 0 }}>
            <strong>4. Synthetic Simulator Regularity:</strong> In the PaySim simulator, fraud transactions strictly emptied accounts in <code>TRANSFER</code> and <code>CASH_OUT</code> channels. Features like <code>origin_balance_error</code> and <code>origin_account_emptied</code> exploit these simulator mechanics directly, whereas real-world fraudsters frequently employ structured partial transfers to evade deterministic heuristic checks.
          </p>
        </div>
      </div>
    </div>
  );
}
