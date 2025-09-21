import React from 'react'

type MultiClassPrediction = {
  prediction: string
  confidence: number
  probabilities: Record<string, number>
}

type EnsemblePredictionProps = {
  ticker: string
  prediction?: MultiClassPrediction
  p?: number // Legacy binary probability
  contribs?: Array<{ key: string; contrib: number }>
  model_info?: {
    version: string
    model_type: string
    ensemble_models?: string[]
  }
}

const CLASS_COLORS = {
  'STRONG_UP': 'text-green-600 bg-green-50',
  'WEAK_UP': 'text-green-500 bg-green-25',
  'SIDEWAYS': 'text-gray-600 bg-gray-50',
  'WEAK_DOWN': 'text-red-500 bg-red-25',
  'STRONG_DOWN': 'text-red-600 bg-red-50'
}

const CLASS_LABELS = {
  'STRONG_UP': 'Strong Buy',
  'WEAK_UP': 'Buy',
  'SIDEWAYS': 'Hold',
  'WEAK_DOWN': 'Sell',
  'STRONG_DOWN': 'Strong Sell'
}

export function EnsemblePrediction({ 
  ticker, 
  prediction, 
  p, 
  contribs = [], 
  model_info 
}: EnsemblePredictionProps) {
  // If we have multiclass prediction, use it; otherwise fall back to binary
  const hasMulticlass = prediction && prediction.probabilities

  return (
    <div className="border rounded-lg p-4 bg-white shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-lg">{ticker}</h3>
        {model_info && (
          <div className="text-xs text-gray-500">
            {model_info.model_type} {model_info.version}
          </div>
        )}
      </div>

      {hasMulticlass ? (
        <div className="space-y-3">
          {/* Main Prediction */}
          <div className="flex items-center space-x-3">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              CLASS_COLORS[prediction.prediction as keyof typeof CLASS_COLORS] || 'text-gray-600 bg-gray-50'
            }`}>
              {CLASS_LABELS[prediction.prediction as keyof typeof CLASS_LABELS] || prediction.prediction}
            </div>
            <div className="text-sm text-gray-600">
              {(prediction.confidence * 100).toFixed(1)}% confidence
            </div>
          </div>

          {/* Probability Distribution */}
          <div className="space-y-1">
            <div className="text-xs font-medium text-gray-700 mb-2">Probability Distribution:</div>
            {Object.entries(prediction.probabilities)
              .sort(([,a], [,b]) => b - a)
              .map(([cls, prob]) => (
                <div key={cls} className="flex items-center space-x-2">
                  <div className="w-20 text-xs text-gray-600">
                    {CLASS_LABELS[cls as keyof typeof CLASS_LABELS] || cls}
                  </div>
                  <div className="flex-1 bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        cls === prediction.prediction ? 'bg-blue-500' : 'bg-gray-400'
                      }`}
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                  <div className="w-12 text-xs text-gray-600 text-right">
                    {(prob * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
          </div>
        </div>
      ) : (
        /* Legacy Binary Prediction */
        <div className="space-y-2">
          <div className="flex items-center space-x-3">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              (p || 0) > 0.6 ? 'text-green-600 bg-green-50' :
              (p || 0) > 0.4 ? 'text-gray-600 bg-gray-50' :
              'text-red-600 bg-red-50'
            }`}>
              {(p || 0) > 0.6 ? 'Buy' : (p || 0) > 0.4 ? 'Hold' : 'Sell'}
            </div>
            <div className="text-sm text-gray-600">
              {((p || 0) * 100).toFixed(1)}% up probability
            </div>
          </div>
        </div>
      )}

      {/* Feature Contributions */}
      {contribs.length > 0 && (
        <div className="mt-4 pt-3 border-t">
          <div className="text-xs font-medium text-gray-700 mb-2">Top Contributing Features:</div>
          <div className="space-y-1">
            {contribs.slice(0, 3).map((contrib, i) => (
              <div key={contrib.key} className="flex items-center justify-between text-xs">
                <span className="text-gray-600">{contrib.key.replace(/_/g, ' ')}</span>
                <span className={`font-medium ${
                  contrib.contrib > 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {contrib.contrib > 0 ? '+' : ''}{contrib.contrib.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Ensemble Model Info */}
      {model_info?.ensemble_models && (
        <div className="mt-3 pt-2 border-t">
          <div className="text-xs text-gray-500">
            Ensemble: {model_info.ensemble_models.join(', ')}
          </div>
        </div>
      )}
    </div>
  )
}

export default EnsemblePrediction
