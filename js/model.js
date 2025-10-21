// js/model.js
export function buildModel(kind, inputLen, lr=0.001){
  const m = tf.sequential();
  if (kind==='cnn1d'){
    m.add(tf.layers.reshape({targetShape:[inputLen,1], inputShape:[inputLen]}));
    m.add(tf.layers.conv1d({filters:32, kernelSize:3, activation:'relu', padding:'same'}));
    m.add(tf.layers.globalAveragePooling1d());
    m.add(tf.layers.dense({units:1, activation:'sigmoid'}));
  } else if (kind==='mlp'){
    m.add(tf.layers.dense({units:128, activation:'relu', inputShape:[inputLen]}));
    m.add(tf.layers.dense({units:64, activation:'relu'}));
    m.add(tf.layers.dense({units:1, activation:'sigmoid'}));
  } else {
    throw new Error('Unknown model kind');
  }
  m.compile({ optimizer: tf.train.adam(lr), loss: 'meanSquaredError', metrics: ['meanAbsoluteError'] });
  return m;
}

export async function fitModel(model, Xtr, ytr, epochs=10, batchSize=256, logFn){
  const xt=tf.tensor2d(Xtr), yt=tf.tensor2d(ytr);
  const h = await model.fit(xt, yt, {
    epochs, batchSize, validationSplit: 0.1,
    callbacks: { onEpochEnd: (ep, logs)=> logFn?.(`ep ${ep+1}/${epochs} loss=${logs.loss.toFixed(6)} val_loss=${(logs.val_loss??0).toFixed(6)} mae=${logs.meanAbsoluteError.toFixed(6)}`) }
  });
  xt.dispose(); yt.dispose();
  return h;
}

export function predictOne(model, xRow){
  const xt = tf.tensor2d([xRow]);
  const y = model.predict(xt);
  const v = y.arraySync()[0][0];
  xt.dispose(); y.dispose?.();
  return v;
}

export function evaluateAccuracy(model, X, y, thr=0.5){
  const xt = tf.tensor2d(X);
  const yp = model.predict(xt);
  const arr = yp.arraySync().map(a=>a[0]);
  xt.dispose(); yp.dispose?.();
  const yTrue = y.map(a=>a[0]);
  let correct=0;
  for (let i=0;i<arr.length;i++){
    const p = arr[i] >= thr ? 1 : 0;
    const t = yTrue[i] >= thr ? 1 : 0;
    if (p===t) correct++;
  }
  return correct/Math.max(1,arr.length);
}
