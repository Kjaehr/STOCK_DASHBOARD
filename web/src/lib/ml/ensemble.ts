import type { EnsembleModel } from './model'
import type { BaseFeatures } from './features'

export type MultiClassPrediction = {
  class: 'STRONG_UP' | 'WEAK_UP' | 'SIDEWAYS' | 'WEAK_DOWN' | 'STRONG_DOWN'
  confidence: number
  probabilities: Record<string, number>
}

export type EnsemblePrediction = {
  prediction: MultiClassPrediction
  feature_contributions: Array<{
    feature: string
    importance: number
    value: number
  }>
  model_info: {
    version: string
    model_type: string
    ensemble_models?: string[]
  }
}

/**
 * Make prediction using ensemble model
 * Since we don't have the actual trained model weights in the JSON,
 * we'll use a heuristic approach based on feature importance and performance metrics
 */
export function predictWithEnsemble(model: EnsembleModel, features: BaseFeatures): EnsemblePrediction {
  // Get feature values for the model's features
  const featureValues: Record<string, number> = {}
  for (const feature of model.features) {
    featureValues[feature] = features[feature] ?? 0
  }

  // Calculate a composite score based on feature importance
  let score = 0
  const contributions: Array<{ feature: string; importance: number; value: number }> = []

  if (model.feature_importance) {
    // Use average importance across all models
    const avgImportance: Record<string, number> = {}
    const modelNames = Object.keys(model.feature_importance)
    
    for (const feature of model.features) {
      let totalImportance = 0
      let count = 0
      
      for (const modelName of modelNames) {
        const importance = model.feature_importance[modelName]?.[feature]
        if (typeof importance === 'number') {
          totalImportance += importance
          count++
        }
      }
      
      avgImportance[feature] = count > 0 ? totalImportance / count : 0
    }

    // Calculate weighted score
    for (const feature of model.features) {
      const value = featureValues[feature]
      const importance = avgImportance[feature] || 0
      const contribution = value * importance
      score += contribution
      
      contributions.push({
        feature,
        importance,
        value
      })
    }
  }

  // Sort contributions by absolute importance
  contributions.sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))

  // Convert score to probabilities using a heuristic approach
  // Based on the model's performance metrics and feature patterns
  const probabilities = scoreToProbabilities(score, model)
  
  // Find the class with highest probability
  const maxClass = Object.entries(probabilities).reduce((max, [cls, prob]) => 
    prob > max.prob ? { class: cls, prob } : max, 
    { class: 'SIDEWAYS', prob: 0 }
  )

  return {
    prediction: {
      class: maxClass.class as MultiClassPrediction['class'],
      confidence: maxClass.prob,
      probabilities
    },
    feature_contributions: contributions.slice(0, 5), // Top 5 features
    model_info: {
      version: model.version,
      model_type: model.model_type,
      ensemble_models: model.model_info?.ensemble_models
    }
  }
}

/**
 * Convert composite score to class probabilities
 * This is a heuristic approach since we don't have the actual model weights
 */
function scoreToProbabilities(score: number, model: EnsembleModel): Record<string, number> {
  // Normalize score to a reasonable range
  const normalizedScore = Math.tanh(score / 10) // Maps to [-1, 1]
  
  // Base probabilities (can be adjusted based on model's label distribution)
  const baseProbabilities = {
    'STRONG_DOWN': 0.2,
    'WEAK_DOWN': 0.2,
    'SIDEWAYS': 0.2,
    'WEAK_UP': 0.2,
    'STRONG_UP': 0.2
  }

  // Adjust probabilities based on score
  const probabilities = { ...baseProbabilities }
  
  if (normalizedScore > 0.3) {
    // Strong positive signal
    probabilities['STRONG_UP'] += 0.3
    probabilities['WEAK_UP'] += 0.2
    probabilities['SIDEWAYS'] -= 0.1
    probabilities['WEAK_DOWN'] -= 0.2
    probabilities['STRONG_DOWN'] -= 0.2
  } else if (normalizedScore > 0.1) {
    // Weak positive signal
    probabilities['WEAK_UP'] += 0.2
    probabilities['STRONG_UP'] += 0.1
    probabilities['SIDEWAYS'] += 0.1
    probabilities['WEAK_DOWN'] -= 0.2
    probabilities['STRONG_DOWN'] -= 0.2
  } else if (normalizedScore < -0.3) {
    // Strong negative signal
    probabilities['STRONG_DOWN'] += 0.3
    probabilities['WEAK_DOWN'] += 0.2
    probabilities['SIDEWAYS'] -= 0.1
    probabilities['WEAK_UP'] -= 0.2
    probabilities['STRONG_UP'] -= 0.2
  } else if (normalizedScore < -0.1) {
    // Weak negative signal
    probabilities['WEAK_DOWN'] += 0.2
    probabilities['STRONG_DOWN'] += 0.1
    probabilities['SIDEWAYS'] += 0.1
    probabilities['WEAK_UP'] -= 0.2
    probabilities['STRONG_UP'] -= 0.2
  } else {
    // Neutral signal
    probabilities['SIDEWAYS'] += 0.2
    probabilities['WEAK_UP'] -= 0.05
    probabilities['WEAK_DOWN'] -= 0.05
    probabilities['STRONG_UP'] -= 0.05
    probabilities['STRONG_DOWN'] -= 0.05
  }

  // Ensure probabilities are non-negative and sum to 1
  const total = Object.values(probabilities).reduce((sum, p) => sum + Math.max(0, p), 0)
  if (total > 0) {
    for (const key in probabilities) {
      const typedKey = key as keyof typeof probabilities
      probabilities[typedKey] = Math.max(0, probabilities[typedKey]) / total
    }
  }

  return probabilities
}

/**
 * Convert ensemble prediction to legacy format for backward compatibility
 */
export function ensemblePredictionToLegacy(prediction: EnsemblePrediction): { p: number; contribs: Array<{ key: string; contrib: number }> } {
  // Convert multi-class to binary probability
  const upProb = (prediction.prediction.probabilities['STRONG_UP'] || 0) + 
                 (prediction.prediction.probabilities['WEAK_UP'] || 0)
  
  const contribs = prediction.feature_contributions.map(fc => ({
    key: fc.feature,
    contrib: fc.importance * fc.value
  }))

  return {
    p: upProb,
    contribs
  }
}
