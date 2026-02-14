/**
 * Confusion matrix heatmap display for training results.
 */

interface ConfusionMatrixProps {
  matrix: number[][];
  labels: string[];
}

export default function ConfusionMatrix({
  matrix,
  labels,
}: ConfusionMatrixProps) {
  const maxVal = Math.max(...matrix.flat(), 1);

  return (
    <div className="bg-sp-card border border-sp-border rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-3">
        Confusion Matrix
      </h3>
      <div className="overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr>
              <th className="p-1 text-gray-500">Pred -&gt;</th>
              {labels.map((l) => (
                <th key={l} className="p-1 text-center text-gray-400 font-mono">
                  {l.slice(0, 4)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={labels[i]}>
                <td className="p-1 text-gray-400 font-mono pr-2">
                  {labels[i].slice(0, 4)}
                </td>
                {row.map((val, j) => {
                  const intensity = val / maxVal;
                  const isCorrect = i === j;
                  return (
                    <td
                      key={j}
                      className="p-1 text-center font-mono rounded"
                      style={{
                        backgroundColor: isCorrect
                          ? `rgba(34, 197, 94, ${intensity * 0.6})`
                          : val > 0
                          ? `rgba(239, 68, 68, ${intensity * 0.6})`
                          : "transparent",
                      }}
                    >
                      {val}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
