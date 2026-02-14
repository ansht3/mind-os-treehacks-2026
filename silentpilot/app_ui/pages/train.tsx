/**
 * Training screen: train the model and view results.
 */

import { useState } from "react";
import Link from "next/link";
import { emgApi } from "../lib/api";
import ConfusionMatrix from "../components/ConfusionMatrix";

const USER_ID = "demo1";

interface TrainResult {
  accuracy: number;
  per_class_accuracy: Record<string, number>;
  confusion_matrix: number[][];
  labels: string[];
  num_samples: number;
}

export default function Train() {
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<TrainResult | null>(null);
  const [error, setError] = useState("");

  const handleTrain = async () => {
    setTraining(true);
    setError("");
    setResult(null);
    try {
      const res = await emgApi.train(USER_ID);
      setResult(res);
    } catch (e) {
      setError(`Training failed: ${e}`);
    }
    setTraining(false);
  };

  return (
    <div className="min-h-screen p-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <Link href="/" className="text-sp-accent text-sm hover:underline">
          &larr; Back
        </Link>
        <h1 className="text-2xl font-bold mt-1">Train Model</h1>
        <p className="text-sm text-gray-400 mt-1">
          User: {USER_ID}
        </p>
      </div>

      {/* Train Button */}
      <div className="flex items-center gap-4 mb-8">
        <button
          onClick={handleTrain}
          disabled={training}
          className="px-8 py-3 bg-sp-accent text-white rounded-lg hover:bg-sp-accent/80 disabled:opacity-50 font-medium"
        >
          {training ? "Training..." : "Train Model"}
        </button>
        {result && (
          <span className="text-sp-green text-sm">
            Model trained successfully!
          </span>
        )}
      </div>

      {error && (
        <div className="bg-sp-red/10 border border-sp-red/30 rounded-lg p-4 mb-6">
          <p className="text-sp-red text-sm">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Summary */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-sp-card border border-sp-border rounded-lg p-4 text-center">
              <p className="text-xs text-gray-500 uppercase mb-1">Accuracy</p>
              <p className="text-3xl font-bold text-sp-green">
                {(result.accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div className="bg-sp-card border border-sp-border rounded-lg p-4 text-center">
              <p className="text-xs text-gray-500 uppercase mb-1">Classes</p>
              <p className="text-3xl font-bold">
                {result.labels.length}
              </p>
            </div>
            <div className="bg-sp-card border border-sp-border rounded-lg p-4 text-center">
              <p className="text-xs text-gray-500 uppercase mb-1">Samples</p>
              <p className="text-3xl font-bold">
                {result.num_samples}
              </p>
            </div>
          </div>

          {/* Per-class accuracy */}
          <div className="bg-sp-card border border-sp-border rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">
              Per-Class Accuracy
            </h3>
            <div className="grid grid-cols-4 gap-3">
              {result.labels.map((label) => {
                const acc = result.per_class_accuracy[label] ?? 0;
                const pct = Math.round(acc * 100);
                return (
                  <div key={label} className="text-center">
                    <p className="text-xs font-mono text-gray-400 mb-1">
                      {label}
                    </p>
                    <p
                      className={`text-lg font-bold ${
                        pct >= 80
                          ? "text-sp-green"
                          : pct >= 60
                          ? "text-sp-yellow"
                          : "text-sp-red"
                      }`}
                    >
                      {pct}%
                    </p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Confusion Matrix */}
          <ConfusionMatrix
            matrix={result.confusion_matrix}
            labels={result.labels}
          />
        </div>
      )}
    </div>
  );
}
