// js/model.js
export function buildModel(kind, inputLength, lr=0.001) {
  // kind: 'mlp' | 'cnn1d' (GRU is disabled in this flat-CSV demo)
  // inputLength: number of flattened features after encoding/scaling
  const model = tf.sequential();

  if (kind === 'mlp') {
    // ---- MLP: simple, strong baseline for tabular data
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [inputLength] }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); // accident_risk in [0,1]
  } else if (kind === 'cnn1d') {
    // ---- CNN-1D: treat features as a 1D signal (length = inputLength, channels = 1)
    // This can sometimes learn local feature interactions better than plain MLP
    model.add(tf.layers.reshape({ targetShape: [inputLength, 1], inputShape: [inputLength] }));
    model.add(tf.layers.conv1d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }));
    model.add(tf.layers.globalAveragePooling1d());
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  } else if (kind === 'gru') {
    // For completeness; disabled in UI since we don't build temporal windows in this demo.
    // If you add sequence windows (T,F), switch to a functional API model here.
    throw new Error('GRU is disabled for flat CSV. Enable only when using (T,F) sequences.');
  } else {
    throw new Error(`Unknown model kind: ${kind}`);
  }

  const optimizer = tf.train.adam(lr);
  model.compile({
    optimizer,
    loss: 'meanSquaredError',                      // regression to [0,1]
    metrics: ['meanAbsoluteError', r2Metric]       // MAE + custom R²
  });
  return model;
}

export async function fitModel(model, Xtrain, ytrain, Xval, yval, {epochs=10, batchSize=256}, logFn) {
  // X*/y* are plain JS arrays. We create/dispose tensors inside.
  const xt = tf.tensor2d(Xtrain);
  const yt = tf.tensor2d(ytrain);
  const xv = tf.tensor2d(Xval);
  const yv = tf.tensor2d(yval);

  const history = await model.fit(xt, yt, {
    epochs, batchSize, validationData: [xv, yv],
    callbacks: {
      onEpochEnd: (ep, logs) => {
        logFn?.(
          `ep ${ep+1}/${epochs}  loss=${logs.loss.toFixed(6)}  val_loss=${(logs.val_loss??0).toFixed(6)}  ` +
          `mae=${logs.meanAbsoluteError.toFixed(6)}  val_mae=${(logs.val_meanAbsoluteError??0).toFixed(6)}  ` +
          `r2=${(logs.r2Metric??0).toFixed(6)}  val_r2=${(logs.val_r2Metric??0).toFixed(6)}`
        );
      }
    }
  });

  xt.dispose(); yt.dispose(); xv.dispose(); yv.dispose();
  return history;
}

export function predict(model, X) {
  const xt = tf.tensor2d(X);
  const yp = model.predict(xt);
  const arr = yp.arraySync(); // [[p],[p],...]
  xt.dispose();
  yp.dispose?.();
  return arr;
}

export async function saveIndexedDB(model, key='accident-risk-model') {
  await model.save(`indexeddb://${key}`);
}

export async function loadIndexedDB(key='accident-risk-model') {
  return await tf.loadLayersModel(`indexeddb://${key}`);
}

export async function exportDownloads(model, filename='accident-risk') {
  await model.save(`downloads://${filename}`); // triggers browser download
}

// ---- Custom R² (coefficient of determination) metric for regression
function r2Metric(yTrue, yPred) {
  const yMean = tf.mean(yTrue);
  const ssTot = tf.sum(tf.pow(tf.sub(yTrue, yMean), 2));
  const ssRes = tf.sum(tf.pow(tf.sub(yTrue, yPred), 2));
  const r2 = tf.sub(1, tf.divNoNan(ssRes, ssTot));
  return r2;
}
