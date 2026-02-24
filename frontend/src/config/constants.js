export const TICKERS = [
  { code: "1120", name: "Al Rajhi Bank",  sector: "Banking",           description: "Largest Islamic bank in Saudi Arabia" },
  { code: "2010", name: "SABIC",          sector: "Petrochemicals",    description: "Saudi Basic Industries Corporation" },
  { code: "7010", name: "STC",            sector: "Telecom",           description: "Saudi Telecom Company" },
  { code: "1150", name: "Alinma Bank",    sector: "Banking",           description: "Islamic banking and financial services" },
  { code: "4325", name: "MASAR",          sector: "Financial Services",description: "MASAR Leasing and Financing Company" },
  { code: "2222", name: "Saudi Aramco",   sector: "Energy",            description: "Largest company on Tadawul by market cap" },
  { code: "1211", name: "Ma'aden",        sector: "Mining",            description: "Saudi Arabian Mining Company" },
  { code: "4110", name: "TAWUNIYA",       sector: "Insurance",         description: "Largest insurance company on Tadawul" },
];

export const DEFAULT_TICKER   = "1120";
export const SECTORS          = [...new Set(TICKERS.map(t => t.sector))];
export const TICKER_MAP       = Object.fromEntries(TICKERS.map(t => [t.code, t]));
export const API_BASE_URL     = process.env.REACT_APP_API_URL || "http://localhost:8000";
