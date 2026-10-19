import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { Network } from 'vis-network/standalone';
import {
  Search,
  RotateCcw,
  Maximize2,
  ZoomIn,
  ZoomOut,
  Sliders,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  ArrowUpRight,
  ArrowDownLeft,
  Copy,
  Check,
  Layers,
  Info,
  ExternalLink,
} from 'lucide-react';
import { useAppContext, formatCurrency, formatNumber } from '../App';

export default function GraphIntelligence() {
  const { API_BASE_URL, graphFocusAccount, setGraphFocusAccount, setActivePage } = useAppContext();
  const containerRef = useRef(null);
  const networkInstanceRef = useRef(null);

  // Exploration controls
  const [searchAccount, setSearchAccount] = useState(graphFocusAccount || '');
  const [hops, setHops] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [graphData, setGraphData] = useState({
    root: null,
    nodes: [],
    edges: [],
    stats: { totalNodes: 0, totalEdges: 0, fraudNodes: 0, fraudEdges: 0, totalVolume: 0 },
    isSample: false,
  });

  // Selected account state
  const [selectedNode, setSelectedNode] = useState(null);
  const [copied, setCopied] = useState(false);

  // Sync if incoming focus account changes
  useEffect(() => {
    if (graphFocusAccount) {
      setSearchAccount(graphFocusAccount);
      loadAccountGraph(graphFocusAccount, hops);
    }
  }, [graphFocusAccount]);

  // Load account-specific subgraph
  const loadAccountGraph = async (accountId, hopCount = 1) => {
    if (!accountId.trim()) return;
    setLoading(true);
    setError('');
    setSelectedNode(null);
    try {
      const res = await fetch(
        `${API_BASE_URL}/graph/account/${encodeURIComponent(accountId.trim())}?hops=${hopCount}&limit=60`
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Failed to explore account: ${res.status}`);
      }
      const data = await res.json();
      setGraphData(data);

      // Auto-select the root node for investigation
      const root = data.nodes.find((n) => n.id === data.root || n.isRoot);
      if (root) setSelectedNode(root);
    } catch (err) {
      setError(err.message || 'Error loading graph network.');
    } finally {
      setLoading(false);
    }
  };

  // Load initial connected network
  const loadInitialNetwork = async () => {
    setLoading(true);
    setError('');
    setSelectedNode(null);
    setSearchAccount('');
    if (setGraphFocusAccount) setGraphFocusAccount(null);
    try {
      const res = await fetch(`${API_BASE_URL}/graph/network?limit=40&include_fraud=true`);
      if (!res.ok) {
        throw new Error(`Failed to load network: ${res.status}`);
      }
      const data = await res.json();
      setGraphData(data);
    } catch (err) {
      setError(err.message || 'Error connecting to graph service.');
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    if (!graphFocusAccount) {
      loadInitialNetwork();
    }
  }, []);

  // Format node for vis-network
  const visNodes = useMemo(() => {
    return graphData.nodes.map((n) => {
      const isRoot = n.id === graphData.root || n.isRoot;
      const isFraud = n.isFraud;

      let bgColor = '#3b82f6'; // normal blue
      let borderColor = '#2563eb';
      let size = 14;

      if (isRoot) {
        bgColor = '#f59e0b'; // amber gold
        borderColor = '#d97706';
        size = 20;
      } else if (isFraud) {
        bgColor = '#ef4444'; // red
        borderColor = '#b91c1c';
        size = 17;
      }

      // Title is tooltip on hover
      const tooltip = `
        <div style="background:#1e293b; color:#f8fafc; padding:8px; border-radius:6px; font-size:12px; border:1px solid #334155;">
          <strong>${n.id}</strong><br/>
          Role: ${isRoot ? 'Focus Account' : isFraud ? 'Fraud Associated' : 'Standard Account'}<br/>
          Inflow: ${n.inDegree} txns (${formatCurrency(n.totalInAmount)})<br/>
          Outflow: ${n.outDegree} txns (${formatCurrency(n.totalOutAmount)})
        </div>
      `;

      return {
        id: n.id,
        label: n.id.length > 8 ? `${n.id.slice(0, 4)}…${n.id.slice(-3)}` : n.id,
        fullLabel: n.id,
        rawNode: n,
        title: tooltip,
        shape: 'dot',
        size,
        color: {
          background: bgColor,
          border: borderColor,
          highlight: {
            background: isFraud ? '#dc2626' : '#60a5fa',
            border: '#ffffff',
          },
          hover: {
            background: '#93c5fd',
            border: '#ffffff',
          },
        },
        borderWidth: isRoot ? 3 : 1.5,
        font: {
          color: '#cbd5e1',
          size: 11,
          face: 'Inter, system-ui, sans-serif',
          vadjust: 3,
        },
      };
    });
  }, [graphData]);

  // Format edges for vis-network
  const visEdges = useMemo(() => {
    const maxAmt = Math.max(...graphData.edges.map((e) => e.amount), 1);

    return graphData.edges.map((e) => {
      const isFraud = e.isFraud;
      const width = Math.max(1.5, Math.min(6, 1.5 + (e.amount / maxAmt) * 4));

      const tooltip = `
        <div style="background:#1e293b; color:#f8fafc; padding:8px; border-radius:6px; font-size:12px; border:1px solid #334155;">
          <strong>Transaction ${e.type}</strong><br/>
          Amount: ${formatCurrency(e.amount)}<br/>
          Step: ${e.step}<br/>
          From: ${e.source}<br/>
          To: ${e.target}<br/>
          Status: ${isFraud ? 'CONFIRMED FRAUD' : 'Legitimate'}
        </div>
      `;

      return {
        id: e.id,
        from: e.source,
        to: e.target,
        title: tooltip,
        arrows: {
          to: {
            enabled: true,
            scaleFactor: 0.65,
          },
        },
        color: {
          color: isFraud ? 'rgba(239, 68, 68, 0.85)' : 'rgba(148, 163, 184, 0.35)',
          highlight: isFraud ? '#ef4444' : '#60a5fa',
          hover: '#93c5fd',
        },
        width,
        smooth: {
          type: 'curvedCW',
          roundness: 0.15,
        },
        rawEdge: e,
      };
    });
  }, [graphData]);

  // Initialize or update vis-network canvas
  useEffect(() => {
    if (!containerRef.current) return;

    if (visNodes.length === 0) {
      if (networkInstanceRef.current) {
        networkInstanceRef.current.destroy();
        networkInstanceRef.current = null;
      }
      return;
    }

    const data = {
      nodes: visNodes,
      edges: visEdges,
    };

    const options = {
      physics: {
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -40,
          centralGravity: 0.015,
          springLength: 95,
          springConstant: 0.08,
          damping: 0.45,
          avoidOverlap: 0.4,
        },
        stabilization: {
          enabled: true,
          iterations: 150,
          updateInterval: 25,
        },
      },
      interaction: {
        hover: true,
        tooltipDelay: 100,
        zoomView: true,
        dragView: true,
        dragNodes: true,
        multiselect: false,
        navigationButtons: false,
        keyboard: false,
      },
    };

    const network = new Network(containerRef.current, data, options);
    networkInstanceRef.current = network;

    // Node click handler
    network.on('click', (params) => {
      if (params.nodes && params.nodes.length > 0) {
        const clickedNodeId = params.nodes[0];
        const fullNode = graphData.nodes.find((n) => n.id === clickedNodeId);
        if (fullNode) {
          setSelectedNode(fullNode);
        }
      }
    });

    return () => {
      if (networkInstanceRef.current) {
        networkInstanceRef.current.destroy();
        networkInstanceRef.current = null;
      }
    };
  }, [visNodes, visEdges, graphData]);

  // Network Controls
  const handleFit = () => {
    if (networkInstanceRef.current) {
      networkInstanceRef.current.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
    }
  };

  const handleZoomIn = () => {
    if (networkInstanceRef.current) {
      const scale = networkInstanceRef.current.getScale();
      networkInstanceRef.current.moveTo({ scale: scale * 1.3, animation: { duration: 300 } });
    }
  };

  const handleZoomOut = () => {
    if (networkInstanceRef.current) {
      const scale = networkInstanceRef.current.getScale();
      networkInstanceRef.current.moveTo({ scale: scale * 0.7, animation: { duration: 300 } });
    }
  };

  const handleCopyAccount = (id) => {
    navigator.clipboard.writeText(id);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  // Connected transactions of selected node
  const selectedNodeTxns = useMemo(() => {
    if (!selectedNode) return [];
    return graphData.edges.filter(
      (e) => e.source === selectedNode.id || e.target === selectedNode.id
    );
  }, [selectedNode, graphData]);

  return (
    <div className="page-enter">
      {/* Header with Title and Search Bar */}
      <div className="explorer-header">
        <div>
          <h2 className="section-title">Graph Intelligence</h2>
          <p className="section-subtitle">
            Force-directed network workspace tracing fund flows, money laundering paths, and ground-truth fraud clusters.
          </p>
        </div>

        {/* Account Search Box */}
        <div className="graph-search-form">
          <div className="input-with-icon">
            <Search size={14} className="field-icon" />
            <input
              type="text"
              placeholder="Search Account ID (e.g. C1305486145)"
              value={searchAccount}
              onChange={(e) => setSearchAccount(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') loadAccountGraph(searchAccount, hops);
              }}
            />
          </div>

          <div className="hops-toggle">
            <button
              className={`btn-hop ${hops === 1 ? 'active' : ''}`}
              onClick={() => {
                setHops(1);
                if (searchAccount) loadAccountGraph(searchAccount, 1);
              }}
              title="1-Hop: Immediate neighbors"
            >
              1-Hop
            </button>
            <button
              className={`btn-hop ${hops === 2 ? 'active' : ''}`}
              onClick={() => {
                setHops(2);
                if (searchAccount) loadAccountGraph(searchAccount, 2);
              }}
              title="2-Hops: Expanded multi-hop network"
            >
              2-Hops
            </button>
          </div>

          <button
            className="btn btn-primary btn-sm"
            onClick={() => loadAccountGraph(searchAccount, hops)}
            disabled={loading || !searchAccount.trim()}
          >
            Explore
          </button>

          <button
            className="btn btn-secondary btn-sm"
            onClick={loadInitialNetwork}
            title="Reset to global fraud cluster overview"
          >
            <RotateCcw size={14} />
            <span>Reset View</span>
          </button>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="notice-banner warning" style={{ marginBottom: '14px' }}>
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Main Workspace Layout */}
      <div className="graph-workspace-grid">
        {/* Network Canvas Card */}
        <div className="graph-canvas-container">
          <div ref={containerRef} className="vis-network-canvas" />

          {/* Canvas Floating Controls */}
          <div className="graph-floating-controls">
            <button className="graph-ctrl-btn" onClick={handleFit} title="Fit entire network into view">
              <Maximize2 size={16} />
            </button>
            <button className="graph-ctrl-btn" onClick={handleZoomIn} title="Zoom in">
              <ZoomIn size={16} />
            </button>
            <button className="graph-ctrl-btn" onClick={handleZoomOut} title="Zoom out">
              <ZoomOut size={16} />
            </button>
          </div>

          {/* Loading Overlay */}
          {loading && (
            <div className="graph-overlay-state">
              <div className="spinner-inline" />
              <span>Constructing network from indexed dataset...</span>
            </div>
          )}

          {/* Empty state */}
          {!loading && visNodes.length === 0 && (
            <div className="graph-overlay-state">
              <Info size={24} className="text-muted" />
              <span>No transactions found for this account. Try another account ID or click Reset View.</span>
            </div>
          )}

          {/* On-Canvas Mini Legend */}
          <div className="graph-canvas-legend">
            <div className="mini-legend-item">
              <span className="mini-dot dot-focus" /> Focus Account
            </div>
            <div className="mini-legend-item">
              <span className="mini-dot dot-fraud" /> Fraud Associated
            </div>
            <div className="mini-legend-item">
              <span className="mini-dot dot-normal" /> Normal Account
            </div>
            <div className="mini-legend-item">
              <span className="mini-line line-fraud" /> Fraud Edge
            </div>
          </div>
        </div>

        {/* Right Sidebar: Investigation Details & Stats */}
        <aside className="graph-inspector-sidebar">
          {/* Selected Account Investigation Panel */}
          {selectedNode ? (
            <div className="inspector-card">
              <div className="inspector-card-header">
                <div className="header-lockup">
                  <span className="text-muted uppercase text-xs">Inspecting Account</span>
                  <div className="account-title-row">
                    <h3 className="font-mono">{selectedNode.id}</h3>
                    <button
                      className="btn-icon"
                      onClick={() => handleCopyAccount(selectedNode.id)}
                      title="Copy Account ID"
                    >
                      {copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}
                    </button>
                  </div>
                </div>

                {selectedNode.isFraud ? (
                  <span className="status-pill status-fraud">
                    <ShieldAlert size={12} /> Fraud Associated
                  </span>
                ) : (
                  <span className="status-pill status-legit">
                    <CheckCircle2 size={12} /> Verified Normal
                  </span>
                )}
              </div>

              {/* Metrics Grid */}
              <div className="inspector-metrics-grid">
                <div className="metric-box">
                  <span className="metric-label">
                    <ArrowDownLeft size={13} className="text-success" /> Total Inflow
                  </span>
                  <span className="metric-val">{formatCurrency(selectedNode.totalInAmount)}</span>
                  <span className="metric-sub">{selectedNode.inDegree} incoming txns</span>
                </div>

                <div className="metric-box">
                  <span className="metric-label">
                    <ArrowUpRight size={13} className="text-danger" /> Total Outflow
                  </span>
                  <span className="metric-val">{formatCurrency(selectedNode.totalOutAmount)}</span>
                  <span className="metric-sub">{selectedNode.outDegree} outgoing txns</span>
                </div>
              </div>

              {/* Connected Transactions List */}
              <div className="inspector-txns-section">
                <h4>Connected Transactions ({selectedNodeTxns.length})</h4>
                <div className="txns-scroll-list">
                  {selectedNodeTxns.map((tx) => (
                    <div
                      key={tx.id}
                      className={`txn-mini-card ${tx.isFraud ? 'txn-mini-fraud' : ''}`}
                    >
                      <div className="txn-mini-top">
                        <span className={`badge-type badge-${tx.type.toLowerCase()}`}>{tx.type}</span>
                        <span className="font-mono font-bold">{formatCurrency(tx.amount)}</span>
                      </div>
                      <div className="txn-mini-flow font-mono text-muted">
                        <span>{tx.source === selectedNode.id ? 'To:' : 'From:'} </span>
                        <span
                          className="sub-account-link"
                          onClick={() => {
                            const other = tx.source === selectedNode.id ? tx.target : tx.source;
                            setSearchAccount(other);
                            loadAccountGraph(other, hops);
                          }}
                        >
                          {tx.source === selectedNode.id ? tx.target : tx.source}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="inspector-actions">
                <button
                  className="btn btn-secondary btn-sm w-full"
                  onClick={() => {
                    setSearchAccount(selectedNode.id);
                    loadAccountGraph(selectedNode.id, hops === 1 ? 2 : 1);
                  }}
                >
                  <Layers size={14} /> Expand Network Depth
                </button>
              </div>
            </div>
          ) : (
            <div className="inspector-card empty-inspector">
              <Info size={28} className="text-muted" />
              <h4>No Account Selected</h4>
              <p className="text-muted text-sm">
                Click any node in the force graph to inspect incoming/outgoing fund flows, degree centrality, and ground-truth fraud evidence.
              </p>
            </div>
          )}

          {/* Network Global Stats */}
          <div className="graph-stats-card">
            <h4>Network Statistics</h4>
            <div className="stat-row">
              <span className="stat-label">Visible Accounts</span>
              <span className="stat-value">{formatNumber(graphData.stats.totalNodes)}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Transactions (Edges)</span>
              <span className="stat-value">{formatNumber(graphData.stats.totalEdges)}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Fraud Accounts</span>
              <span className="stat-value fraud">{formatNumber(graphData.stats.fraudNodes)}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Fraud Transactions</span>
              <span className="stat-value fraud">{formatNumber(graphData.stats.fraudEdges)}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Total Flow Volume</span>
              <span className="stat-value">{formatCurrency(graphData.stats.totalVolume)}</span>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
