import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { Check, ChevronDown, ChevronRight, Clock, Copy, Download, Loader2, Play, Save, Trash2, TrendingDown, TrendingUp, Minus, BarChart3, Users, Target, Bell, BellRing, CheckCircle2, XCircle, RefreshCw, Plus, FileText, Info, HelpCircle } from 'lucide-react';
import { useEffect, useState, useRef } from 'react';

// Info Tooltip Component
function InfoTooltip({ text }: { text: string }) {
  const [show, setShow] = useState(false);
  return (
    <div className="relative inline-block">
      <HelpCircle
        size={14}
        className="text-gray-500 hover:text-gray-300 cursor-help ml-1"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
      />
      {show && (
        <div className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 text-xs text-white bg-gray-800 border border-gray-700 rounded-lg shadow-lg whitespace-nowrap max-w-xs">
          {text}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-800" />
        </div>
      )}
    </div>
  );
}

interface AgentAnalysis {
  agent: string;
  signal: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  confidence: number;
  reasoning: string;
}

interface StockAnalysis {
  ticker: string;
  agents: AgentAnalysis[];
}

interface PortfolioItem {
  ticker: string;
  action: string;
  quantity: number;
  confidence: number;
  bullish: number;
  bearish: number;
  neutral: number;
}

interface HistoryItem {
  id: number;
  name: string;
  timestamp: string;
  tickers: string[];
}

interface Analyst {
  key: string;
  display_name: string;
  description: string;
}

interface PerformanceData {
  ticker: string;
  signal: string;
  current_price: number;
  price_change: number;
  price_change_percent: number;
  signal_correct: boolean;
}

interface AlertItem {
  id: number;
  name: string;
  condition_type: string;
  condition_value: any;
  is_active: boolean;
  trigger_count: number;
}

interface TriggeredAlert {
  id: number;
  name: string;
  message: string;
  ticker?: string;
}

function parseMarkdown(markdown: string): { stocks: StockAnalysis[]; portfolio: PortfolioItem[]; strategy: string } {
  const stocks: StockAnalysis[] = [];
  const portfolio: PortfolioItem[] = [];
  let strategy = '';

  const stockSections = markdown.split(/Analysis for (\w+)\n={50,}/);

  for (let i = 1; i < stockSections.length; i += 2) {
    const ticker = stockSections[i];
    const content = stockSections[i + 1] || '';
    const agents: AgentAnalysis[] = [];
    const tableRows = content.match(/\|\s*([^|]+?)\s*\|\s*(BULLISH|BEARISH|NEUTRAL)\s*\|\s*([\d.]+)%?\s*\|([^|]+)\|/g);

    if (tableRows) {
      for (const row of tableRows) {
        const match = row.match(/\|\s*([^|]+?)\s*\|\s*(BULLISH|BEARISH|NEUTRAL)\s*\|\s*([\d.]+)%?\s*\|([^|]+)\|/);
        if (match) {
          const agentName = match[1].trim();
          if (agentName && agentName !== 'Agent' && !agentName.includes('===')) {
            agents.push({
              agent: agentName,
              signal: match[2] as 'BULLISH' | 'BEARISH' | 'NEUTRAL',
              confidence: parseFloat(match[3]),
              reasoning: match[4].trim(),
            });
          }
        }
      }
    }

    if (agents.length > 0) {
      stocks.push({ ticker, agents });
    }
  }

  const portfolioMatch = markdown.match(/PORTFOLIO SUMMARY:[\s\S]*?\+[-+]+\+([\s\S]*?)\n\nPortfolio Strategy:/);
  if (portfolioMatch) {
    const portfolioRows = portfolioMatch[1].match(/\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)%?\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|/g);
    if (portfolioRows) {
      for (const row of portfolioRows) {
        const match = row.match(/\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)%?\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|/);
        if (match && match[1] !== 'Ticker') {
          portfolio.push({
            ticker: match[1],
            action: match[2],
            quantity: parseInt(match[3]),
            confidence: parseFloat(match[4]),
            bullish: parseInt(match[5]),
            bearish: parseInt(match[6]),
            neutral: parseInt(match[7]),
          });
        }
      }
    }
  }

  const strategyMatch = markdown.match(/Portfolio Strategy:\s*(.+)/);
  if (strategyMatch) {
    strategy = strategyMatch[1].trim();
  }

  return { stocks, portfolio, strategy };
}

// Performance Tracking Component
function PerformanceTracker({ stocks }: { stocks: StockAnalysis[] }) {
  const [performance, setPerformance] = useState<PerformanceData[]>([]);
  const [loading, setLoading] = useState(false);

  const checkPerformance = async () => {
    setLoading(true);
    try {
      const signals = stocks.map(stock => {
        const bullish = stock.agents.filter(a => a.signal === 'BULLISH').length;
        const bearish = stock.agents.filter(a => a.signal === 'BEARISH').length;
        let signal = 'NEUTRAL';
        if (bearish > bullish) signal = 'BEARISH';
        else if (bullish > bearish) signal = 'BULLISH';
        return { ticker: stock.ticker, signal, signal_date: new Date().toISOString() };
      });

      const res = await fetch('http://localhost:8000/stock-data/check-performance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ signals }),
      });
      if (res.ok) setPerformance(await res.json());
    } catch (err) {
      console.error('Failed to check performance:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="border border-gray-700 rounded-lg p-4 bg-gray-800/20">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-blue-400" />
          <h3 className="font-medium text-sm text-white">Performance Check</h3>
          <InfoTooltip text="Verifies if AI signals were correct by comparing with actual stock price movements" />
        </div>
        <Button variant="outline" size="sm" onClick={checkPerformance} disabled={loading} className="h-7 text-xs gap-1">
          {loading ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
          Check
        </Button>
      </div>

      {performance.length === 0 ? (
        <p className="text-xs text-gray-500 text-center py-2">Click Check to verify signals</p>
      ) : (
        <div className="space-y-2">
          {performance.map((p) => (
            <div key={p.ticker} className={cn("flex items-center justify-between p-2 rounded text-sm", p.signal_correct ? "bg-green-900/20" : "bg-red-900/20")}>
              <div className="flex items-center gap-2">
                {p.signal_correct ? <CheckCircle2 className="w-4 h-4 text-green-500" /> : <XCircle className="w-4 h-4 text-red-500" />}
                <span className="text-white font-medium">{p.ticker}</span>
              </div>
              <span className={cn("font-medium", p.price_change >= 0 ? "text-green-400" : "text-red-400")}>
                {p.price_change >= 0 ? '+' : ''}{p.price_change_percent.toFixed(1)}%
              </span>
            </div>
          ))}
          <div className="text-xs text-gray-400 text-center pt-1">
            Accuracy: {((performance.filter(p => p.signal_correct).length / performance.length) * 100).toFixed(0)}%
          </div>
        </div>
      )}
    </div>
  );
}

// Alerts Component
function AlertsManager({ stocks }: { stocks: StockAnalysis[] }) {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [triggered, setTriggered] = useState<TriggeredAlert[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [newType, setNewType] = useState<'consensus'>('consensus');
  const [newSignal, setNewSignal] = useState<'BEARISH'>('BEARISH');
  const [newValue, setNewValue] = useState(5);

  const fetchAlerts = async () => {
    try {
      const res = await fetch('http://localhost:8000/alerts/');
      if (res.ok) setAlerts(await res.json());
    } catch (err) { console.error(err); }
  };

  const checkAlerts = async () => {
    try {
      const res = await fetch('http://localhost:8000/alerts/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stocks, portfolio: [] }),
      });
      if (res.ok) setTriggered(await res.json());
    } catch (err) { console.error(err); }
  };

  const createAlert = async () => {
    if (!newName.trim()) return;
    await fetch('http://localhost:8000/alerts/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: newName,
        condition: { type: newType, params: { signal: newSignal, min_count: newValue } },
      }),
    });
    setShowCreate(false);
    setNewName('');
    fetchAlerts();
  };

  const deleteAlert = async (id: number) => {
    await fetch(`http://localhost:8000/alerts/${id}`, { method: 'DELETE' });
    fetchAlerts();
  };

  useEffect(() => { fetchAlerts(); }, []);
  useEffect(() => { if (stocks.length && alerts.length) checkAlerts(); }, [stocks, alerts]);

  return (
    <div className="border border-gray-700 rounded-lg p-4 bg-gray-800/20">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Bell className="w-4 h-4 text-yellow-400" />
          <h3 className="font-medium text-sm text-white">Alerts</h3>
          <InfoTooltip text="Get notified when multiple agents agree on a signal (e.g., 5+ agents are bearish)" />
          {triggered.length > 0 && <span className="px-1.5 py-0.5 rounded-full bg-red-500 text-white text-xs">{triggered.length}</span>}
        </div>
        <Button variant="outline" size="sm" onClick={() => setShowCreate(true)} className="h-7 text-xs gap-1">
          <Plus size={12} />Add
        </Button>
      </div>

      {triggered.length > 0 && (
        <div className="mb-3 space-y-1">
          {triggered.map((t, i) => (
            <div key={i} className="flex items-center gap-2 p-2 rounded bg-yellow-900/20 text-xs">
              <BellRing className="w-3 h-3 text-yellow-400" />
              <span className="text-yellow-200">{t.message}</span>
            </div>
          ))}
        </div>
      )}

      {showCreate && (
        <div className="mb-3 p-2 rounded bg-gray-800 space-y-2">
          <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Alert name" className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs text-white" />
          <div className="flex gap-2">
            <select value={newSignal} onChange={(e) => setNewSignal(e.target.value as any)} className="flex-1 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs text-white">
              <option value="BEARISH">Bearish</option>
              <option value="BULLISH">Bullish</option>
            </select>
            <input type="number" value={newValue} onChange={(e) => setNewValue(Number(e.target.value))} className="w-16 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs text-white" />
          </div>
          <div className="flex gap-2">
            <Button size="sm" onClick={createAlert} className="h-6 text-xs">Create</Button>
            <Button variant="ghost" size="sm" onClick={() => setShowCreate(false)} className="h-6 text-xs">Cancel</Button>
          </div>
        </div>
      )}

      <div className="space-y-1">
        {alerts.length === 0 ? (
          <p className="text-xs text-gray-500 text-center">No alerts</p>
        ) : alerts.slice(0, 3).map((a) => (
          <div key={a.id} className="flex items-center justify-between p-1.5 rounded bg-gray-800/50 text-xs">
            <span className="text-gray-300 truncate">{a.name}</span>
            <button onClick={() => deleteAlert(a.id)} className="text-gray-500 hover:text-red-400"><Trash2 size={12} /></button>
          </div>
        ))}
      </div>
    </div>
  );
}

// Cleaner Portfolio Card
function PortfolioCard({ item }: { item: PortfolioItem }) {
  const isShort = item.action === 'SHORT';
  const total = item.bullish + item.bearish + item.neutral;

  return (
    <div className={cn("rounded-lg p-4 border", isShort ? "bg-red-950/30 border-red-800/50" : "bg-green-950/30 border-green-800/50")}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-2xl font-bold text-white">{item.ticker}</span>
          <span className={cn("px-2 py-0.5 rounded text-xs font-bold", isShort ? "bg-red-600 text-white" : "bg-green-600 text-white")}>
            {item.action}
          </span>
        </div>
        {isShort ? <TrendingDown className="w-5 h-5 text-red-400" /> : <TrendingUp className="w-5 h-5 text-green-400" />}
      </div>

      <div className="mb-3">
        <div className="text-xs text-gray-400">Position</div>
        <div className="text-xl font-bold text-white">{item.quantity} shares</div>
      </div>

      <div className="mb-3">
        <div className="flex justify-between text-xs mb-1">
          <span className="text-gray-400">Confidence</span>
          <span className="text-white font-medium">{item.confidence.toFixed(0)}%</span>
        </div>
        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
          <div className={cn("h-full", item.confidence >= 70 ? "bg-green-500" : item.confidence >= 40 ? "bg-yellow-500" : "bg-red-500")} style={{ width: `${item.confidence}%` }} />
        </div>
      </div>

      <div className="flex gap-1 h-5">
        {item.bullish > 0 && <div className="bg-green-600 rounded flex items-center justify-center text-white text-xs font-medium px-2" style={{ flex: item.bullish }}>{item.bullish}</div>}
        {item.neutral > 0 && <div className="bg-gray-600 rounded flex items-center justify-center text-white text-xs font-medium px-2" style={{ flex: item.neutral }}>{item.neutral}</div>}
        {item.bearish > 0 && <div className="bg-red-600 rounded flex items-center justify-center text-white text-xs font-medium px-2" style={{ flex: item.bearish }}>{item.bearish}</div>}
      </div>
    </div>
  );
}

// Agent Leaderboard
function AgentLeaderboard({ stocks }: { stocks: StockAnalysis[] }) {
  const agentStats: Record<string, { bullish: number; bearish: number; neutral: number; total: number; confidence: number }> = {};
  stocks.forEach(stock => {
    stock.agents.forEach(agent => {
      if (!agentStats[agent.agent]) agentStats[agent.agent] = { bullish: 0, bearish: 0, neutral: 0, total: 0, confidence: 0 };
      agentStats[agent.agent][agent.signal.toLowerCase() as 'bullish' | 'bearish' | 'neutral']++;
      agentStats[agent.agent].total++;
      agentStats[agent.agent].confidence += agent.confidence;
    });
  });

  const sorted = Object.entries(agentStats).map(([name, s]) => ({ name, ...s, avg: s.confidence / s.total })).sort((a, b) => b.avg - a.avg);

  return (
    <div className="border border-gray-700 rounded-lg p-4 bg-gray-800/20">
      <div className="flex items-center gap-2 mb-3">
        <Users className="w-4 h-4 text-purple-400" />
        <h3 className="font-medium text-sm text-white">Top Agents</h3>
        <InfoTooltip text="Ranks AI agents by their average confidence level across all analyzed stocks" />
      </div>
      <div className="space-y-1">
        {sorted.slice(0, 5).map((a, i) => (
          <div key={a.name} className="flex items-center gap-2 p-1.5 rounded bg-gray-800/50 text-xs">
            <span className={cn("w-5 h-5 rounded-full flex items-center justify-center font-bold", i === 0 ? "bg-yellow-500 text-black" : "bg-gray-700 text-gray-300")}>{i + 1}</span>
            <span className="flex-1 text-white truncate">{a.name}</span>
            <span className="text-blue-400 font-medium">{a.avg.toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Signal Summary
function SignalSummary({ portfolio }: { portfolio: PortfolioItem[] }) {
  const bull = portfolio.reduce((s, p) => s + p.bullish, 0);
  const bear = portfolio.reduce((s, p) => s + p.bearish, 0);
  const neut = portfolio.reduce((s, p) => s + p.neutral, 0);
  const total = bull + bear + neut;

  return (
    <div className="border border-gray-700 rounded-lg p-4 bg-gray-800/20">
      <div className="flex items-center gap-2 mb-3">
        <BarChart3 className="w-4 h-4 text-blue-400" />
        <h3 className="font-medium text-sm text-white">Signal Summary</h3>
        <InfoTooltip text="Total count of bullish, neutral, and bearish signals from all AI agents across all stocks" />
      </div>
      <div className="flex gap-2 mb-3">
        <div className="flex-1 text-center p-2 rounded bg-green-900/30">
          <div className="text-lg font-bold text-green-400">{bull}</div>
          <div className="text-xs text-gray-400">Bull</div>
        </div>
        <div className="flex-1 text-center p-2 rounded bg-gray-800/50">
          <div className="text-lg font-bold text-gray-400">{neut}</div>
          <div className="text-xs text-gray-400">Neutral</div>
        </div>
        <div className="flex-1 text-center p-2 rounded bg-red-900/30">
          <div className="text-lg font-bold text-red-400">{bear}</div>
          <div className="text-xs text-gray-400">Bear</div>
        </div>
      </div>
      <div className="h-3 rounded-full overflow-hidden flex">
        <div className="bg-green-500" style={{ width: `${(bull / total) * 100}%` }} />
        <div className="bg-gray-500" style={{ width: `${(neut / total) * 100}%` }} />
        <div className="bg-red-500" style={{ width: `${(bear / total) * 100}%` }} />
      </div>
    </div>
  );
}

function SignalBadge({ signal }: { signal: 'BULLISH' | 'BEARISH' | 'NEUTRAL' }) {
  if (signal === 'BULLISH') return <span className="px-2 py-0.5 text-xs rounded-full bg-green-600/30 text-green-400">Bullish</span>;
  if (signal === 'BEARISH') return <span className="px-2 py-0.5 text-xs rounded-full bg-red-600/30 text-red-400">Bearish</span>;
  return <span className="px-2 py-0.5 text-xs rounded-full bg-gray-600/30 text-gray-400">Neutral</span>;
}

function StockCard({ stock, portfolio }: { stock: StockAnalysis; portfolio?: PortfolioItem }) {
  const [expanded, setExpanded] = useState(false);
  const bull = stock.agents.filter(a => a.signal === 'BULLISH').length;
  const bear = stock.agents.filter(a => a.signal === 'BEARISH').length;
  const neut = stock.agents.filter(a => a.signal === 'NEUTRAL').length;

  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden bg-gray-800/20">
      <div className="p-3 cursor-pointer hover:bg-gray-700/20" onClick={() => setExpanded(!expanded)}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <span className="text-lg font-bold text-white">{stock.ticker}</span>
            {portfolio && <span className={cn("px-2 py-0.5 rounded text-xs font-medium", portfolio.action === 'SHORT' ? "bg-red-600 text-white" : "bg-green-600 text-white")}>{portfolio.action}</span>}
          </div>
          <div className="flex gap-1 text-xs">
            <span className="px-2 py-0.5 rounded bg-green-600/30 text-green-400">{bull}</span>
            <span className="px-2 py-0.5 rounded bg-gray-600/30 text-gray-400">{neut}</span>
            <span className="px-2 py-0.5 rounded bg-red-600/30 text-red-400">{bear}</span>
          </div>
        </div>
      </div>
      {expanded && (
        <div className="border-t border-gray-700 p-3 space-y-2">
          {stock.agents.map((agent, i) => (
            <div key={i} className="flex items-start gap-3 p-2 rounded bg-gray-800/30 text-sm">
              <div className="w-32 flex-shrink-0">
                <div className="font-medium text-white text-xs">{agent.agent}</div>
                <SignalBadge signal={agent.signal} />
              </div>
              <div className="flex-1">
                <div className="text-xs text-gray-400 mb-1">Confidence: {agent.confidence.toFixed(0)}%</div>
                <p className="text-xs text-gray-500 line-clamp-2">{agent.reasoning}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// PDF Export function
function generatePDF(data: { stocks: StockAnalysis[]; portfolio: PortfolioItem[]; strategy: string }) {
  const content = `
AI HEDGE FUND ANALYSIS REPORT
Generated: ${new Date().toLocaleString()}
${'='.repeat(50)}

PORTFOLIO RECOMMENDATIONS
${'-'.repeat(30)}
${data.portfolio.map(p => `
${p.ticker}: ${p.action} ${p.quantity} shares
Confidence: ${p.confidence.toFixed(0)}%
Votes: ${p.bullish} Bullish | ${p.neutral} Neutral | ${p.bearish} Bearish
`).join('\n')}

Strategy: ${data.strategy}

${'='.repeat(50)}
DETAILED ANALYSIS
${'-'.repeat(30)}
${data.stocks.map(stock => `
${stock.ticker}
${'-'.repeat(20)}
${stock.agents.map(a => `
  ${a.agent}
  Signal: ${a.signal} | Confidence: ${a.confidence.toFixed(0)}%
  ${a.reasoning.substring(0, 200)}...
`).join('\n')}`).join('\n\n')}
`;

  // Create a printable HTML document
  const printWindow = window.open('', '_blank');
  if (printWindow) {
    printWindow.document.write(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>AI Hedge Fund Analysis Report</title>
        <style>
          body { font-family: Arial, sans-serif; padding: 40px; max-width: 800px; margin: 0 auto; }
          h1 { color: #1a1a1a; border-bottom: 2px solid #333; padding-bottom: 10px; }
          h2 { color: #333; margin-top: 30px; }
          .portfolio-item { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 8px; }
          .ticker { font-size: 24px; font-weight: bold; }
          .action-buy { color: #16a34a; }
          .action-sell { color: #dc2626; }
          .agent { background: #fff; padding: 10px; margin: 5px 0; border-left: 3px solid #666; }
          .bullish { border-left-color: #16a34a; }
          .bearish { border-left-color: #dc2626; }
          .neutral { border-left-color: #666; }
          @media print { body { padding: 20px; } }
        </style>
      </head>
      <body>
        <h1>AI Hedge Fund Analysis Report</h1>
        <p>Generated: ${new Date().toLocaleString()}</p>

        <h2>Portfolio Recommendations</h2>
        ${data.portfolio.map(p => `
          <div class="portfolio-item">
            <span class="ticker">${p.ticker}</span>
            <span class="action-${p.action === 'SHORT' ? 'sell' : 'buy'}">${p.action} ${p.quantity} shares</span>
            <div>Confidence: ${p.confidence.toFixed(0)}%</div>
            <div>Votes: ${p.bullish} Bullish | ${p.neutral} Neutral | ${p.bearish} Bearish</div>
          </div>
        `).join('')}

        <p><strong>Strategy:</strong> ${data.strategy}</p>

        <h2>Detailed Analysis</h2>
        ${data.stocks.map(stock => `
          <h3>${stock.ticker}</h3>
          ${stock.agents.map(a => `
            <div class="agent ${a.signal.toLowerCase()}">
              <strong>${a.agent}</strong> - ${a.signal} (${a.confidence.toFixed(0)}%)
              <p style="font-size: 12px; color: #666;">${a.reasoning.substring(0, 300)}...</p>
            </div>
          `).join('')}
        `).join('')}

        <script>window.print();</script>
      </body>
      </html>
    `);
    printWindow.document.close();
  }
}

export function Project1Results({ className }: { className?: string }) {
  const [markdown, setMarkdown] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [data, setData] = useState<{ stocks: StockAnalysis[]; portfolio: PortfolioItem[]; strategy: string } | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [activeTab, setActiveTab] = useState<'results' | 'run'>('results');
  const [analysts, setAnalysts] = useState<Analyst[]>([]);
  const [tickerInput, setTickerInput] = useState('AAPL, MSFT, NVDA');
  const [selectedAnalysts, setSelectedAnalysts] = useState<Set<string>>(new Set());
  const [isRunning, setIsRunning] = useState(false);
  const [runProgress, setRunProgress] = useState<string[]>([]);

  const fetchHistory = async () => {
    try {
      const res = await fetch('http://localhost:8000/run-history/');
      if (res.ok) setHistory(await res.json());
    } catch (err) { console.error(err); }
  };

  const fetchAnalysts = async () => {
    try {
      const res = await fetch('http://localhost:8000/hedge-fund/agents');
      if (res.ok) {
        const d = await res.json();
        setAnalysts(d.agents || []);
        setSelectedAnalysts(new Set(d.agents.slice(0, 6).map((a: Analyst) => a.key)));
      }
    } catch (err) { console.error(err); }
  };

  const loadCurrentFile = async () => {
    try {
      const res = await fetch('/project1_run.md');
      if (!res.ok) throw new Error('No results file found');
      const text = await res.text();
      setMarkdown(text);
      setData(parseMarkdown(text));
      setSelectedRunId(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error');
    } finally { setIsLoading(false); }
  };

  const loadRun = async (runId: number) => {
    setIsLoading(true);
    try {
      const res = await fetch(`http://localhost:8000/run-history/${runId}`);
      if (!res.ok) throw new Error('Failed to load');
      const d = await res.json();
      setMarkdown(d.markdown_content);
      setData(parseMarkdown(d.markdown_content));
      setSelectedRunId(runId);
      setShowHistory(false);
      setActiveTab('results');
    } catch (err) { setError(err instanceof Error ? err.message : 'Error'); }
    finally { setIsLoading(false); }
  };

  const saveToHistory = async () => {
    if (!saveName.trim()) return;
    await fetch('http://localhost:8000/run-history/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: saveName, tickers: data?.stocks.map(s => s.ticker) || [], markdown_content: markdown }),
    });
    setShowSaveDialog(false);
    setSaveName('');
    fetchHistory();
  };

  const deleteRun = async (runId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete?')) return;
    await fetch(`http://localhost:8000/run-history/${runId}`, { method: 'DELETE' });
    fetchHistory();
    if (selectedRunId === runId) loadCurrentFile();
  };

  // Run analysis function using simple-analysis endpoint
  const runAnalysis = async () => {
    if (selectedAnalysts.size === 0) {
      alert('Please select at least one analyst');
      return;
    }

    const tickers = tickerInput.split(',').map(t => t.trim().toUpperCase()).filter(t => t);
    if (tickers.length === 0) {
      alert('Please enter at least one ticker');
      return;
    }

    setIsRunning(true);
    setRunProgress([]);
    setRunProgress(prev => [...prev, `Starting analysis for ${tickers.join(', ')}...`]);

    try {
      // Clear any previous run
      await fetch('http://localhost:8000/simple-analysis/clear', { method: 'POST' });

      // Start the analysis
      const startRes = await fetch('http://localhost:8000/simple-analysis/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tickers,
          analysts: Array.from(selectedAnalysts),
          model_name: 'gpt-4o-mini',
        }),
      });

      if (!startRes.ok) {
        const err = await startRes.json();
        throw new Error(err.detail || 'Failed to start analysis');
      }

      // Poll for status
      let lastProgressLength = 0;
      const pollStatus = async (): Promise<any> => {
        const statusRes = await fetch('http://localhost:8000/simple-analysis/status');
        const status = await statusRes.json();

        // Update progress messages
        if (status.progress && status.progress.length > lastProgressLength) {
          const newMessages = status.progress.slice(lastProgressLength);
          setRunProgress(prev => [...prev, ...newMessages]);
          lastProgressLength = status.progress.length;
        }

        if (status.status === 'complete') {
          return status.result;
        } else if (status.status === 'error') {
          throw new Error(status.error || 'Analysis failed');
        } else {
          // Still running, poll again
          await new Promise(resolve => setTimeout(resolve, 1000));
          return pollStatus();
        }
      };

      const result = await pollStatus();

      if (result) {
        // Convert result to markdown format
        const md = convertResultToMarkdown(result, tickers);
        setMarkdown(md);
        setData(parseMarkdown(md));
        setActiveTab('results');
        setRunProgress(prev => [...prev, 'Analysis complete! Switching to results...']);

        // Save to history
        await fetch('http://localhost:8000/run-history/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: `Run ${new Date().toLocaleString()}`,
            tickers,
            markdown_content: md,
          }),
        });
        fetchHistory();
      }
    } catch (err) {
      setRunProgress(prev => [...prev, `Error: ${err instanceof Error ? err.message : 'Unknown error'}`]);
    } finally {
      setIsRunning(false);
    }
  };

  // Convert API result to markdown
  const convertResultToMarkdown = (result: any, tickers: string[]): string => {
    let md = '';
    const signals = result.analyst_signals || {};
    const decisions = result.decisions?.decisions || result.decisions || [];

    // Use provided tickers or extract from result
    const tickerList = tickers.length > 0 ? tickers : (result.tickers || Object.keys(signals));

    for (const ticker of tickerList) {
      md += `Analysis for ${ticker}\n${'='.repeat(50)}\n\n`;
      md += `AGENT ANALYSIS: [${ticker}]\n`;
      md += '+' + '-'.repeat(23) + '+' + '-'.repeat(10) + '+' + '-'.repeat(14) + '+' + '-'.repeat(62) + '+\n';
      md += '| Agent                 |  Signal  |   Confidence | Reasoning                                                    |\n';
      md += '+' + '='.repeat(23) + '+' + '='.repeat(10) + '+' + '='.repeat(14) + '+' + '='.repeat(62) + '+\n';

      const tickerSignals = signals[ticker] || {};
      for (const [agentKey, signal] of Object.entries(tickerSignals) as [string, any][]) {
        const agentName = agentKey.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()).substring(0, 21).padEnd(21);
        const sig = (signal.signal || 'NEUTRAL').toUpperCase().padEnd(8);
        const conf = `${(signal.confidence || 0).toFixed(0)}%`.padStart(12);
        const reason = (typeof signal.reasoning === 'string' ? signal.reasoning : JSON.stringify(signal.reasoning || '')).substring(0, 60).padEnd(60);
        md += `| ${agentName} | ${sig} | ${conf} | ${reason} |\n`;
        md += '+' + '-'.repeat(23) + '+' + '-'.repeat(10) + '+' + '-'.repeat(14) + '+' + '-'.repeat(62) + '+\n';
      }
      md += '\n';
    }

    md += 'PORTFOLIO SUMMARY:\n';
    md += '+' + '-'.repeat(10) + '+' + '-'.repeat(10) + '+' + '-'.repeat(12) + '+' + '-'.repeat(14) + '+' + '-'.repeat(11) + '+' + '-'.repeat(11) + '+' + '-'.repeat(11) + '+\n';
    md += '| Ticker   |  Action  |   Quantity |   Confidence |  Bullish  |  Bearish  |  Neutral  |\n';
    md += '+' + '='.repeat(10) + '+' + '='.repeat(10) + '+' + '='.repeat(12) + '+' + '='.repeat(14) + '+' + '='.repeat(11) + '+' + '='.repeat(11) + '+' + '='.repeat(11) + '+\n';

    // If no decisions, create from signals
    const decisionList = Array.isArray(decisions) && decisions.length > 0
      ? decisions
      : tickerList.map(ticker => {
          const tickerSignals = signals[ticker] || {};
          let bull = 0, bear = 0;
          for (const sig of Object.values(tickerSignals) as any[]) {
            if (sig.signal?.toUpperCase() === 'BULLISH') bull++;
            else if (sig.signal?.toUpperCase() === 'BEARISH') bear++;
          }
          return {
            ticker,
            action: bear > bull ? 'short' : bull > bear ? 'buy' : 'hold',
            quantity: Math.abs(bear - bull) * 10,
            confidence: Math.max(bull, bear) / (bull + bear + 1) * 100,
          };
        });

    for (const dec of decisionList) {
      const tickerSignals = signals[dec.ticker] || {};
      let bull = 0, bear = 0, neut = 0;
      for (const sig of Object.values(tickerSignals) as any[]) {
        if (sig.signal?.toUpperCase() === 'BULLISH') bull++;
        else if (sig.signal?.toUpperCase() === 'BEARISH') bear++;
        else neut++;
      }
      const action = (dec.action || 'hold').toUpperCase().padEnd(8);
      md += `| ${dec.ticker.padEnd(8)} | ${action} | ${String(dec.quantity || 0).padStart(10)} | ${((dec.confidence || 0).toFixed(1) + '%').padStart(12)} | ${String(bull).padStart(9)} | ${String(bear).padStart(9)} | ${String(neut).padStart(9)} |\n`;
      md += '+' + '-'.repeat(10) + '+' + '-'.repeat(10) + '+' + '-'.repeat(12) + '+' + '-'.repeat(14) + '+' + '-'.repeat(11) + '+' + '-'.repeat(11) + '+' + '-'.repeat(11) + '+\n';
    }

    md += '\nPortfolio Strategy:\nAI-generated trading recommendations\n';
    return md;
  };

  useEffect(() => {
    loadCurrentFile();
    fetchHistory();
    fetchAnalysts();
  }, []);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(markdown);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const selectedRun = history.find(h => h.id === selectedRunId);

  if (isLoading) return <div className={cn("h-full w-full flex items-center justify-center bg-gray-900", className)}><Loader2 className="animate-spin text-gray-400" /></div>;

  return (
    <div className={cn("h-full w-full flex bg-gray-900", className)}>
      {/* Sidebar */}
      <div className={cn("border-r border-gray-800 bg-gray-900 transition-all overflow-hidden flex flex-col", showHistory ? "w-56" : "w-0")}>
        <div className="p-3 border-b border-gray-800 flex items-center justify-between">
          <span className="font-medium text-sm text-white">History</span>
          <button onClick={() => setShowHistory(false)} className="text-gray-500 hover:text-white"><ChevronRight size={14} /></button>
        </div>
        <div className="flex-1 overflow-auto p-2 space-y-1">
          <button onClick={loadCurrentFile} className={cn("w-full text-left p-2 rounded text-sm", selectedRunId === null ? "bg-blue-600/20 text-blue-400" : "hover:bg-gray-800 text-gray-400")}>
            <div className="font-medium">Latest</div>
          </button>
          {history.map((item) => (
            <button key={item.id} onClick={() => loadRun(item.id)} className={cn("w-full text-left p-2 rounded text-sm group relative", selectedRunId === item.id ? "bg-blue-600/20 text-blue-400" : "hover:bg-gray-800 text-gray-400")}>
              <div className="truncate pr-5">{item.name}</div>
              <div className="text-xs text-gray-600">{new Date(item.timestamp).toLocaleDateString()}</div>
              <button onClick={(e) => deleteRun(item.id, e)} className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 text-red-500"><Trash2 size={12} /></button>
            </button>
          ))}
        </div>
      </div>

      {/* Main */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="border-b border-gray-800 bg-gray-900 px-4 py-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {!showHistory && <button onClick={() => setShowHistory(true)} className="text-gray-500 hover:text-white text-sm flex items-center gap-1"><Clock size={14} />History</button>}
              <div className="flex bg-gray-800 rounded p-0.5">
                <button onClick={() => setActiveTab('results')} className={cn("px-3 py-1 rounded text-sm", activeTab === 'results' ? "bg-blue-600 text-white" : "text-gray-400")}>Results</button>
                <button onClick={() => setActiveTab('run')} className={cn("px-3 py-1 rounded text-sm", activeTab === 'run' ? "bg-blue-600 text-white" : "text-gray-400")}>Run</button>
              </div>
            </div>
            {activeTab === 'results' && data && (
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={() => setShowSaveDialog(true)} className="h-7 text-xs"><Save size={12} className="mr-1" />Save</Button>
                <Button variant="outline" size="sm" onClick={handleCopy} className="h-7 text-xs">{copied ? <Check size={12} /> : <Copy size={12} />}</Button>
                <Button variant="outline" size="sm" onClick={() => data && generatePDF(data)} className="h-7 text-xs"><FileText size={12} className="mr-1" />PDF</Button>
              </div>
            )}
          </div>
        </div>

        {showSaveDialog && (
          <div className="p-3 border-b border-gray-800 bg-gray-800/50 flex items-center gap-2">
            <input value={saveName} onChange={(e) => setSaveName(e.target.value)} placeholder="Name..." className="flex-1 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white" autoFocus />
            <Button size="sm" onClick={saveToHistory} className="h-7">Save</Button>
            <Button variant="ghost" size="sm" onClick={() => setShowSaveDialog(false)} className="h-7">Cancel</Button>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-auto">
          {activeTab === 'results' ? (
            error ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-500">
                <p className="mb-4">No analysis results yet</p>
                <Button onClick={() => setActiveTab('run')}>Run Analysis</Button>
              </div>
            ) : (
              <div className="p-4 space-y-4">
                {/* Portfolio */}
                {data && data.portfolio.length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <h2 className="text-base font-semibold text-white">Recommendations</h2>
                      <InfoTooltip text="Trading actions based on aggregate AI agent analysis - BUY/SHORT with position size and confidence" />
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      {data.portfolio.map((item) => <PortfolioCard key={item.ticker} item={item} />)}
                    </div>
                  </div>
                )}

                {/* Stats - 2 columns */}
                {data && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <SignalSummary portfolio={data.portfolio} />
                    <AgentLeaderboard stocks={data.stocks} />
                  </div>
                )}

                {/* Performance & Alerts - 2 columns */}
                {data && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <PerformanceTracker stocks={data.stocks} />
                    <AlertsManager stocks={data.stocks} />
                  </div>
                )}

                {/* Details */}
                {data && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <h2 className="text-base font-semibold text-white">Details</h2>
                      <InfoTooltip text="Expand each stock to see individual AI agent signals, confidence levels, and reasoning" />
                    </div>
                    {data.stocks.map((stock) => <StockCard key={stock.ticker} stock={stock} portfolio={data.portfolio.find(p => p.ticker === stock.ticker)} />)}
                  </div>
                )}
              </div>
            )
          ) : (
            /* Run Tab */
            <div className="p-4 max-w-2xl mx-auto space-y-4">
              <div className="text-center">
                <h2 className="text-xl font-bold text-white mb-1">Run Analysis</h2>
                <p className="text-sm text-gray-500">Enter tickers and select agents</p>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-1">Tickers</label>
                <input value={tickerInput} onChange={(e) => setTickerInput(e.target.value)} placeholder="AAPL, MSFT, NVDA" disabled={isRunning}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white" />
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-sm text-gray-400">Agents ({selectedAnalysts.size})</label>
                  <div className="flex gap-2 text-xs">
                    <button onClick={() => setSelectedAnalysts(new Set(analysts.map(a => a.key)))} className="text-blue-400">All</button>
                    <button onClick={() => setSelectedAnalysts(new Set())} className="text-gray-500">Clear</button>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-1.5 max-h-48 overflow-auto">
                  {analysts.map((a) => (
                    <button key={a.key} onClick={() => {
                      const s = new Set(selectedAnalysts);
                      s.has(a.key) ? s.delete(a.key) : s.add(a.key);
                      setSelectedAnalysts(s);
                    }} disabled={isRunning}
                      className={cn("p-2 rounded border text-left text-xs", selectedAnalysts.has(a.key) ? "border-blue-500 bg-blue-900/20 text-white" : "border-gray-700 text-gray-500")}>
                      {a.display_name}
                    </button>
                  ))}
                </div>
              </div>

              <Button onClick={runAnalysis} disabled={isRunning} className="w-full py-4">
                {isRunning ? <><Loader2 className="animate-spin mr-2" />Running...</> : <><Play className="mr-2" />Run Analysis</>}
              </Button>

              {runProgress.length > 0 && (
                <div className="bg-gray-800 rounded p-3 max-h-40 overflow-auto">
                  {runProgress.map((p, i) => (
                    <div key={i} className={cn("text-xs py-0.5", p.includes('Error') ? "text-red-400" : p.includes('complete') ? "text-green-400" : "text-gray-400")}>{p}</div>
                  ))}
                </div>
              )}

              {/* Show View Results button after analysis completes */}
              {!isRunning && data && runProgress.some(p => p.includes('complete')) && (
                <Button variant="outline" onClick={() => setActiveTab('results')} className="w-full mt-2">
                  <FileText className="mr-2" size={16} />
                  View Results
                </Button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
