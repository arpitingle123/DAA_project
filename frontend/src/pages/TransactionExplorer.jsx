import React, { useState, useEffect, useCallback } from "react";
import {
  Search,
  Filter,
  Download,
  RotateCcw,
  Network,
  ChevronLeft,
  ChevronRight,
  Eye,
  X,
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  ArrowUpRight,
  ArrowDownLeft,
  SlidersHorizontal,
} from "lucide-react";
import { useAppContext, formatCurrency, formatNumber } from "../App";

export default function TransactionExplorer() {
  const { API_BASE_URL, navigateToGraph } = useAppContext();

  // Search & Filter State (Immediate input values)
  const [txIdInput, setTxIdInput] = useState("");
  const [senderInput, setSenderInput] = useState("");
  const [receiverInput, setReceiverInput] = useState("");
  const [minAmountInput, setMinAmountInput] = useState("");
  const [maxAmountInput, setMaxAmountInput] = useState("");
  const [stepInput, setStepInput] = useState("");

  // Debounced filters applied to queries
  const [txId, setTxId] = useState("");
  const [sender, setSender] = useState("");
  const [receiver, setReceiver] = useState("");
  const [type, setType] = useState("");
  const [isFraud, setIsFraud] = useState("");
  const [minAmount, setMinAmount] = useState("");
  const [maxAmount, setMaxAmount] = useState("");
  const [step, setStep] = useState("");
  const [sortBy, setSortBy] = useState("id");
  const [sortDir, setSortDir] = useState("asc");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);

  // Debounce text inputs by 350ms to prevent excessive API requests
  useEffect(() => {
    const timer = setTimeout(() => {
      setTxId(txIdInput.trim());
      setSender(senderInput.trim());
      setReceiver(receiverInput.trim());
      setMinAmount(minAmountInput.trim());
      setMaxAmount(maxAmountInput.trim());
      setStep(stepInput.trim());
      setPage(1);
    }, 350);
    return () => clearTimeout(timer);
  }, [txIdInput, senderInput, receiverInput, minAmountInput, maxAmountInput, stepInput]);

  // Data & UI State
  const [data, setData] = useState({
    total: 0,
    page: 1,
    page_size: 20,
    total_pages: 1,
    items: [],
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedTx, setSelectedTx] = useState(null);
  const [inspectingModel, setInspectingModel] = useState(false);
  const [modelResult, setModelResult] = useState(null);

  // Fetch transactions from backend
  const fetchTransactions = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({
        page: String(page),
        page_size: String(pageSize),
        sort_by: sortBy,
        sort_dir: sortDir,
      });

      if (txId) params.append("transaction_id", txId);
      if (sender) params.append("sender", sender);
      if (receiver) params.append("receiver", receiver);
      if (type) params.append("type", type);
      if (isFraud !== "") params.append("is_fraud", isFraud);
      if (minAmount) params.append("min_amount", minAmount);
      if (maxAmount) params.append("max_amount", maxAmount);
      if (step) params.append("step", step);

      const resp = await fetch(`${API_BASE_URL}/transactions?${params.toString()}`);
      if (!resp.ok) {
        const errJson = await resp.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server error: ${resp.status}`);
      }
      const json = await resp.json();
      setData(json);
    } catch (err) {
      setError(err.message || "Failed to load transactions.");
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL, page, pageSize, sortBy, sortDir, txId, sender, receiver, type, isFraud, minAmount, maxAmount, step]);

  useEffect(() => {
    fetchTransactions();
  }, [fetchTransactions]);

  // Handle column header sort
  const handleSort = (col) => {
    if (sortBy === col) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(col);
      setSortDir("asc");
    }
    setPage(1);
  };

  const handleResetFilters = () => {
    setTxIdInput("");
    setSenderInput("");
    setReceiverInput("");
    setMinAmountInput("");
    setMaxAmountInput("");
    setStepInput("");
    setTxId("");
    setSender("");
    setReceiver("");
    setType("");
    setIsFraud("");
    setMinAmount("");
    setMaxAmount("");
    setStep("");
    setSortBy("id");
    setSortDir("asc");
    setPage(1);
  };

  // Export filtered rows to CSV
  const handleExportCSV = () => {
    if (!data.items || data.items.length === 0) return;
    const headers = [
      "ID",
      "Step",
      "Type",
      "Amount",
      "Sender_nameOrig",
      "OldBalance_Orig",
      "NewBalance_Orig",
      "Receiver_nameDest",
      "OldBalance_Dest",
      "NewBalance_Dest",
      "GroundTruth_isFraud",
      "isFlaggedFraud",
    ];

    const rows = data.items.map((it) => [
      it.id,
      it.step,
      it.type,
      it.amount,
      it.nameOrig,
      it.oldbalanceOrg,
      it.newbalanceOrig,
      it.nameDest,
      it.oldbalanceDest,
      it.newbalanceDest,
      it.isFraud,
      it.isFlaggedFraud,
    ]);

    const csvContent =
      "data:text/csv;charset=utf-8," +
      [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute(
      "download",
      `fingraph_transactions_page${page}_${new Date().toISOString().slice(0, 10)}.csv`
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Run live model inference on selected transaction
  const handleRunModelInference = async (tx) => {
    setInspectingModel(true);
    setModelResult(null);
    try {
      const payload = {
        step: tx.step,
        type: tx.type,
        amount: tx.amount,
        oldbalanceOrg: tx.oldbalanceOrg,
        newbalanceOrig: tx.newbalanceOrig,
        oldbalanceDest: tx.oldbalanceDest,
        newbalanceDest: tx.newbalanceDest,
        nameOrig: tx.nameOrig,
        nameDest: tx.nameDest,
        model_preference: "random_forest",
      };
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("Inference failed");
      const resJson = await res.json();
      setModelResult(resJson);
    } catch (e) {
      setModelResult({ error: e.message });
    } finally {
      setInspectingModel(false);
    }
  };

  return (
    <div className="page-enter">
      <div className="explorer-header">
        <div>
          <h2 className="section-title">Transaction Explorer</h2>
          <p className="section-subtitle">
            Search, filter, and inspect verified records across all {formatNumber(data.total || 6362620)} dataset transactions.
          </p>
        </div>
        <div className="explorer-actions">
          <button
            className="btn btn-secondary btn-sm"
            onClick={handleExportCSV}
            disabled={loading || data.items.length === 0}
            title="Download current page records as CSV"
          >
            <Download size={14} />
            <span>Export CSV</span>
          </button>
          <button
            className="btn btn-secondary btn-sm"
            onClick={handleResetFilters}
            title="Reset all filters"
          >
            <RotateCcw size={14} />
            <span>Reset Filters</span>
          </button>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="explorer-filter-card">
        <div className="filter-grid">
          <div className="filter-field">
            <label>Transaction ID (Row #)</label>
            <div className="input-with-icon">
              <Search size={14} className="field-icon" />
              <input
                type="number"
                min="1"
                placeholder="e.g. 1042"
                value={txIdInput}
                onChange={(e) => setTxIdInput(e.target.value)}
              />
            </div>
          </div>

          <div className="filter-field">
            <label>Sender Account (nameOrig)</label>
            <div className="input-with-icon">
              <Search size={14} className="field-icon" />
              <input
                type="text"
                placeholder="e.g. C1231006815"
                value={senderInput}
                onChange={(e) => setSenderInput(e.target.value)}
              />
            </div>
          </div>

          <div className="filter-field">
            <label>Receiver Account (nameDest)</label>
            <div className="input-with-icon">
              <Search size={14} className="field-icon" />
              <input
                type="text"
                placeholder="e.g. M1979787155"
                value={receiverInput}
                onChange={(e) => setReceiverInput(e.target.value)}
              />
            </div>
          </div>

          <div className="filter-field">
            <label>Transaction Type</label>
            <select
              value={type}
              onChange={(e) => {
                setType(e.target.value);
                setPage(1);
              }}
            >
              <option value="">All Types</option>
              <option value="TRANSFER">TRANSFER</option>
              <option value="CASH_OUT">CASH_OUT</option>
              <option value="PAYMENT">PAYMENT</option>
              <option value="CASH_IN">CASH_IN</option>
              <option value="DEBIT">DEBIT</option>
            </select>
          </div>

          <div className="filter-field">
            <label>Ground-Truth Fraud</label>
            <select
              value={isFraud}
              onChange={(e) => {
                setIsFraud(e.target.value);
                setPage(1);
              }}
            >
              <option value="">All Transactions</option>
              <option value="1">Fraud Only (isFraud = 1)</option>
              <option value="0">Legitimate Only (isFraud = 0)</option>
            </select>
          </div>

          <div className="filter-field">
            <label>Min Amount ($)</label>
            <input
              type="number"
              placeholder="Min $"
              min="0"
              value={minAmountInput}
              onChange={(e) => setMinAmountInput(e.target.value)}
            />
          </div>

          <div className="filter-field">
            <label>Max Amount ($)</label>
            <input
              type="number"
              placeholder="Max $"
              min="0"
              value={maxAmountInput}
              onChange={(e) => setMaxAmountInput(e.target.value)}
            />
          </div>

          <div className="filter-field">
            <label>Step (Hour)</label>
            <input
              type="number"
              placeholder="e.g. 1"
              min="0"
              value={stepInput}
              onChange={(e) => setStepInput(e.target.value)}
            />
          </div>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="notice-banner warning" style={{ marginBottom: "16px" }}>
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Main Table Card */}
      <div className="card table-card">
        <div className="table-meta-bar">
          <span className="results-count">
            {loading ? (
              "Loading records..."
            ) : (
              <>
                Showing <strong>{data.items.length > 0 ? (page - 1) * pageSize + 1 : 0}</strong>–
                <strong>{Math.min(page * pageSize, data.total)}</strong> of{" "}
                <strong>{formatNumber(data.total)}</strong> matching records
              </>
            )}
          </span>

          <div className="page-size-selector">
            <span>Per page:</span>
            <select
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setPage(1);
              }}
            >
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>
        </div>

        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th onClick={() => handleSort("id")} className="sortable">
                  Row ID {sortBy === "id" && (sortDir === "asc" ? "▲" : "▼")}
                </th>
                <th onClick={() => handleSort("step")} className="sortable">
                  Step {sortBy === "step" && (sortDir === "asc" ? "▲" : "▼")}
                </th>
                <th onClick={() => handleSort("type")} className="sortable">
                  Type {sortBy === "type" && (sortDir === "asc" ? "▲" : "▼")}
                </th>
                <th onClick={() => handleSort("amount")} className="sortable text-right">
                  Amount {sortBy === "amount" && (sortDir === "asc" ? "▲" : "▼")}
                </th>
                <th>Sender (Origin)</th>
                <th>Receiver (Dest)</th>
                <th className="text-right">Origin Balance</th>
                <th className="text-center">Dataset Label</th>
                <th className="text-center">Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading && data.items.length === 0 ? (
                <tr>
                  <td colSpan={9} className="table-loading">
                    <div className="spinner-inline" /> Searching transactions...
                  </td>
                </tr>
              ) : data.items.length === 0 ? (
                <tr>
                  <td colSpan={9} className="table-empty">
                    No transactions match the selected filters.
                  </td>
                </tr>
              ) : (
                data.items.map((tx) => (
                  <tr key={tx.id} className={tx.isFraud ? "row-fraud" : ""}>
                    <td className="font-mono text-muted">#{tx.id}</td>
                    <td className="font-mono">{tx.step}</td>
                    <td>
                      <span className={`badge-type badge-${tx.type.toLowerCase()}`}>
                        {tx.type}
                      </span>
                    </td>
                    <td className="text-right font-mono font-bold">
                      {formatCurrency(tx.amount)}
                    </td>
                    <td>
                      <span
                        className="account-link"
                        onClick={() => navigateToGraph(tx.nameOrig)}
                        title="Explore sender in Graph Intelligence"
                      >
                        {tx.nameOrig}
                      </span>
                    </td>
                    <td>
                      <span
                        className="account-link"
                        onClick={() => navigateToGraph(tx.nameDest)}
                        title="Explore receiver in Graph Intelligence"
                      >
                        {tx.nameDest}
                      </span>
                    </td>
                    <td className="text-right font-mono text-secondary">
                      {formatCurrency(tx.oldbalanceOrg)}
                    </td>
                    <td className="text-center">
                      {tx.isFraud === 1 ? (
                        <span className="status-pill status-fraud">
                          <ShieldAlert size={12} /> FRAUD
                        </span>
                      ) : (
                        <span className="status-pill status-legit">
                          <CheckCircle2 size={12} /> Legitimate
                        </span>
                      )}
                    </td>
                    <td className="text-center actions-cell">
                      <button
                        className="btn-icon"
                        onClick={() => {
                          setSelectedTx(tx);
                          setModelResult(null);
                        }}
                        title="Inspect transaction details"
                      >
                        <Eye size={15} />
                      </button>
                      <button
                        className="btn-icon btn-icon-accent"
                        onClick={() => navigateToGraph(tx.nameOrig)}
                        title="Investigate Sender in Graph Intelligence"
                      >
                        <Network size={15} />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        <div className="table-pagination">
          <div className="pagination-info">
            Page {data.page} of {data.total_pages || 1}
          </div>

          <div className="pagination-controls">
            <button
              className="btn btn-secondary btn-sm"
              disabled={page <= 1 || loading}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              <ChevronLeft size={16} /> Previous
            </button>
            <span className="page-pill">{page}</span>
            <button
              className="btn btn-secondary btn-sm"
              disabled={page >= data.total_pages || loading}
              onClick={() => setPage((p) => Math.min(data.total_pages, p + 1))}
            >
              Next <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Transaction Details Modal / Drawer */}
      {selectedTx && (
        <div className="modal-overlay" onClick={() => setSelectedTx(null)}>
          <div className="modal-card modal-lg" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-lockup">
                <h3>Transaction Record #{selectedTx.id}</h3>
                <span className={`badge-type badge-${selectedTx.type.toLowerCase()}`}>
                  {selectedTx.type}
                </span>
                {selectedTx.isFraud === 1 ? (
                  <span className="status-pill status-fraud">
                    <ShieldAlert size={13} /> Ground-Truth Fraud
                  </span>
                ) : (
                  <span className="status-pill status-legit">
                    <CheckCircle2 size={13} /> Ground-Truth Legitimate
                  </span>
                )}
              </div>
              <button className="btn-close" onClick={() => setSelectedTx(null)}>
                <X size={18} />
              </button>
            </div>

            <div className="modal-body">
              {/* Financial Summary Grid */}
              <div className="detail-stat-grid">
                <div className="detail-stat-box">
                  <span className="stat-label">Amount Transacted</span>
                  <span className="stat-val highlight">{formatCurrency(selectedTx.amount)}</span>
                </div>
                <div className="detail-stat-box">
                  <span className="stat-label">Time Step (Hour)</span>
                  <span className="stat-val">Hour {selectedTx.step}</span>
                </div>
                <div className="detail-stat-box">
                  <span className="stat-label">Origin Discrepancy</span>
                  <span className="stat-val font-mono">
                    {formatCurrency(
                      Math.abs(selectedTx.oldbalanceOrg - selectedTx.amount - selectedTx.newbalanceOrig)
                    )}
                  </span>
                </div>
                <div className="detail-stat-box">
                  <span className="stat-label">Destination Discrepancy</span>
                  <span className="stat-val font-mono">
                    {formatCurrency(
                      Math.abs(selectedTx.oldbalanceDest + selectedTx.amount - selectedTx.newbalanceDest)
                    )}
                  </span>
                </div>
              </div>

              {/* Account Pair Details */}
              <div className="account-pair-grid">
                <div className="account-col">
                  <h4>
                    <ArrowUpRight size={16} className="text-danger" /> Sender Account (Origin)
                  </h4>
                  <div className="account-info-box">
                    <div className="account-id-row">
                      <span className="font-mono">{selectedTx.nameOrig}</span>
                      <button
                        className="btn btn-xs btn-secondary"
                        onClick={() => {
                          setSelectedTx(null);
                          navigateToGraph(selectedTx.nameOrig);
                        }}
                      >
                        <Network size={12} /> View Graph
                      </button>
                    </div>
                    <div className="balance-track">
                      <div>
                        <span className="sub-label">Old Balance:</span>
                        <span className="font-mono">{formatCurrency(selectedTx.oldbalanceOrg)}</span>
                      </div>
                      <div className="arrow-sep">→</div>
                      <div>
                        <span className="sub-label">New Balance:</span>
                        <span className="font-mono">{formatCurrency(selectedTx.newbalanceOrig)}</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="account-col">
                  <h4>
                    <ArrowDownLeft size={16} className="text-success" /> Receiver Account (Destination)
                  </h4>
                  <div className="account-info-box">
                    <div className="account-id-row">
                      <span className="font-mono">{selectedTx.nameDest}</span>
                      <button
                        className="btn btn-xs btn-secondary"
                        onClick={() => {
                          setSelectedTx(null);
                          navigateToGraph(selectedTx.nameDest);
                        }}
                      >
                        <Network size={12} /> View Graph
                      </button>
                    </div>
                    <div className="balance-track">
                      <div>
                        <span className="sub-label">Old Balance:</span>
                        <span className="font-mono">{formatCurrency(selectedTx.oldbalanceDest)}</span>
                      </div>
                      <div className="arrow-sep">→</div>
                      <div>
                        <span className="sub-label">New Balance:</span>
                        <span className="font-mono">{formatCurrency(selectedTx.newbalanceDest)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Explainability Distinction */}
              <div className="explainability-callout">
                <h5>Explainability & Evidence Breakdown</h5>
                <ul>
                  <li>
                    <strong>Dataset Ground Truth:</strong> This transaction is recorded as{" "}
                    <strong>{selectedTx.isFraud ? "CONFIRMED FRAUD (isFraud = 1)" : "LEGITIMATE (isFraud = 0)"}</strong> in PaySim.
                  </li>
                  <li>
                    <strong>Origin Account Emptied:</strong>{" "}
                    {selectedTx.oldbalanceOrg > 0 && selectedTx.newbalanceOrig === 0
                      ? "Yes — Account was completely emptied after transaction."
                      : "No."}
                  </li>
                  <li>
                    <strong>Transfer / Cashout Channel:</strong>{" "}
                    {["TRANSFER", "CASH_OUT"].includes(selectedTx.type)
                      ? "High-risk transfer/cashout channel (primary vehicle for synthetic fraud)."
                      : "Standard retail payment / deposit channel."}
                  </li>
                </ul>
              </div>

              {/* Model Verification Block */}
              <div className="model-verify-block">
                <div className="model-verify-header">
                  <h5>Live Random Forest Prediction Verification</h5>
                  <button
                    className="btn btn-sm btn-primary"
                    disabled={inspectingModel}
                    onClick={() => handleRunModelInference(selectedTx)}
                  >
                    {inspectingModel ? "Running Model..." : "Evaluate with Random Forest"}
                  </button>
                </div>

                {modelResult && (
                  <div className="model-eval-results">
                    {modelResult.error ? (
                      <div className="notice-banner warning">
                        <AlertTriangle size={14} />
                        <span>{modelResult.error}</span>
                      </div>
                    ) : (
                      <div className="eval-metrics-row">
                        <div>
                          <span className="sub-label">Model Probability:</span>
                          <span className="eval-val">
                            {(modelResult.fraud_probability * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div>
                          <span className="sub-label">Predicted Label:</span>
                          <span className="eval-val">
                            {modelResult.predicted_is_fraud === 1 ? "FRAUD (1)" : "LEGITIMATE (0)"}
                          </span>
                        </div>
                        <div>
                          <span className="sub-label">Risk Category:</span>
                          <span className={`eval-val risk-${modelResult.risk_level.toLowerCase()}`}>
                            {modelResult.risk_level}
                          </span>
                        </div>
                        <div>
                          <span className="sub-label">Model Used:</span>
                          <span className="eval-val">{modelResult.model_used}</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="modal-footer">
              <button
                className="btn btn-secondary"
                onClick={() => setSelectedTx(null)}
              >
                Close
              </button>
              <button
                className="btn btn-primary"
                onClick={() => {
                  setSelectedTx(null);
                  navigateToGraph(selectedTx.nameOrig);
                }}
              >
                <Network size={14} /> Open Sender in Graph Intelligence
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
