export type SentimentLabel = "positive" | "neutral" | "negative";

export type AnalysisStatus = "pending" | "processing" | "completed" | "failed";

export interface OverviewMetrics {
  total_feedback: number;
  total_units: number;
  positive: number;
  neutral: number;
  negative: number;
  positive_percentage: number;
  neutral_percentage: number;
  negative_percentage: number;
}

export interface TopicMetric {
  topic_id: number;
  label: string;
  unit_count: number;
  sentiment: {
    positive: number;
    neutral: number;
    negative: number;
  };
}

export interface DashboardResponse {
  analysis: {
    id: string;
    name: string;
    status: AnalysisStatus;
  };
  overview: OverviewMetrics;
  topics: TopicMetric[];
  top_issues: Array<{text: string; frequency: number; topic_id: number | null}>;
  trend: Array<{
    date: string;
    total_units: number;
    sentiment: {
      positive: number;
      neutral: number;
      negative: number;
    };
  }>;
  insight: {
    status: "pending" | "completed" | "failed" | "skipped";
    summary: string | null;
  };
}
