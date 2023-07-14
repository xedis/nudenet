const fs = require('fs');
const { ExifTool } = require("exiftool-vendored");
const path = require('path');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');
const argparse = require('argparse');

const { createCanvas, loadImage } = require('canvas');

const options = {
  debug: true,
  modelPath: 'file://models/default-f16/model.json',
  minScore: 0.38,
  maxResults: 50,
  iouThreshold: 0.5,
  outputNodes: ['output1', 'output2', 'output3'],
  blurNude: false,
  blurRadius: 25,
  labels: undefined,
  classes: {
    base: [
      'exposed belly',
      'exposed buttocks',
      'exposed breasts',
      'exposed vagina',
      'exposed penis',
      'male breast',
    ],
    default: [
      'exposed anus',
      'exposed armpits',
      'belly',
      'exposed belly',
      'buttocks',
      'exposed buttocks',
      'female face',
      'male face',
      'feet',
      'exposed feet',
      'breast',
      'exposed breast',
      'vagina',
      'exposed vagina',
      'male breast',
      'exposed penis',
    ],
  },
  composite: 'default',
  composites: {
    base: {
      person: [],
      sexy: [],
      nude: [2, 3, 4],
    },
    default: {
      person: [6, 7],
      sexy: [1, 2, 3, 4, 8, 9, 10, 15],
      nude: [0, 5, 11, 12, 13],
    },
  },
};

const models = {};

function rect({ canvas, x = 0, y = 0, width = 0, height = 0, radius = 8, lineWidth = 2, color = 'white', title = '', font = '16px "Segoe UI"' }) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.lineWidth = 2;
  ctx.fillStyle = color;
  ctx.font = font;
  ctx.fillText(title, x + 4, y - 4);
}

function getTensorFromImage(imageFile) {
  if (!fs.existsSync(imageFile)) {
    log.error('Not found:', imageFile);
    return null;
  }
  const data = fs.readFileSync(imageFile);
  const bufferT = tf.node.decodeImage(data, 3);
  const expandedT = tf.expandDims(bufferT, 0);
  const imageT = tf.cast(expandedT, 'float32');
  imageT.file = imageFile;
  tf.dispose([expandedT, bufferT]);
  if (options.debug) log.info('loaded image:', imageT.file, 'width:', imageT.shape[2], 'height:', imageT.shape[1]);
  return imageT;
}

async function saveProcessedImage(inImage, outImage, data) {
  if (!data) return false;
  return new Promise(async (resolve) => {
    const original = await loadImage(inImage);
    const canvas = createCanvas(original.width, original.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(original, 0, 0, canvas.width, canvas.height);
    for (const obj of data.parts) {
      if (options.composite[options.composite].includes(obj.id) && options.blurNude) {
        blur({ canvas, left: obj.box[0], top: obj.box[1], width: obj.box[2], height: obj.box[3] });
      }
      rect({ canvas, x: obj.box[0], y: obj.box[1], width: obj.box[2], height: obj.box[3], title: `${Math.round(100 * obj.score)}% ${obj.class}` });
    }
    const tag = bewbs(data);
    const out = fs.createWriteStream(outImage);
    out.on('finish', () => {
      if (options.debug) log.state('created output image:', outImage);
      resolve(true);
    });
    out.on('error', (err) => {
      log.error('error creating image:', outImage, err);
      resolve(true);
    });
    const stream = canvas.createJPEGStream({ quality: 0.6, progressive: true, chromaSubsampling: true });
    stream.pipe(out);
  });
}

function bewbs(data) {
  return (data.nude === true || data.nude === 'true') ? 'NSFW' : 'SFW';
}

async function processPrediction(boxesTensor, scoresTensor, classesTensor, inputTensor) {
  const boxes = await boxesTensor.array();
  const scores = await scoresTensor.data();
  const classes = await classesTensor.data();
  const nmsT = await tf.image.nonMaxSuppressionAsync(boxes[0], scores, options.maxResults, options.iouThreshold, options.minScore);
  const nms = await nmsT.data();
  tf.dispose(nmsT);
  const parts = [];
  for (const i in nms) {
    const id = parseInt(i);
    parts.push({
      score: scores[i],
      id: classes[id],
      class: options.labels[classes[id]],
      box: [
        Math.trunc(boxes[0][id][0]),
        Math.trunc(boxes[0][id][1]),
        Math.trunc((boxes[0][id][3] - boxes[0][id][1])),
        Math.trunc((boxes[0][id][2] - boxes[0][id][0])),
      ],
    });
  }
  const result = {
    input: { file: inputTensor.file, width: inputTensor.shape[2], height: inputTensor.shape[1] },
    person: parts.filter((a) => options.composites[options.composite].person.includes(a.id)).length > 0,
    sexy: parts.filter((a) => options.composites[options.composite].sexy.includes(a.id)).length > 0,
    nude: parts.filter((a) => options.composites[options.composite].nude.includes(a.id)).length > 0,
    parts,
  };
  if (options.debug) log.data('result:', result);
  return result;
}

async function runDetection(input, output) {
  if (!models[options.modelPath]) {
    try {
      models[options.modelPath] = await tf.loadGraphModel(options.modelPath);
      models[options.modelPath].path = options.modelPath;
      if (options.debug) log.state('loaded graph model:', options.modelPath);
      if (models[options.modelPath].version === 'v2.base') {
        options.labels = options.classes.base;
        options.composite = 'base';
      } else {
        options.labels = options.classes.default;
        options.composite = 'default';
      }
    } catch (err) {
      log.error('error loading graph model:', options.modelPath, err.message, err);
      return null;
    }
  }
  const t = {};
  t.input = getTensorFromImage(input);
  [t.boxes, t.scores, t.classes] = await models[options.modelPath].executeAsync(t.input, options.outputNodes);
  const res = await processPrediction(t.boxes, t.scores, t.classes, t.input);
  if (output) await saveProcessedImage(input, output, res);
  Object.keys(t).forEach((tensor) => tf.dispose(t[tensor]));
  return res;
}

async function addMetadata(inImage, tag) {
  const et = new ExifTool();
  try {
    await et.write(inImage, { 'PNGWarning': tag });
    console.log(`Wrote ${tag} to ${inImage}`);
  } catch (err) {
    console.error(`Failed to write ${tag} to ${inImage}:`, err);
  } finally {
    await et.end();
  }
}

async function main() {
  log.header();
  const parser = new argparse.ArgumentParser({ description: 'nudenet' });
  parser.add_argument('--model', '-m', { type: 'str', required: false, default: null, help: 'path to model' });
  parser.add_argument('--input', '-i', { type: 'str', required: true, default: null, help: 'input image' });
  parser.add_argument('--output', '-o', { type: 'str', required: false, default: null, help: 'output image' });
  const args = parser.parse_args();
  if (args.model) {
    if (fs.existsSync(args.model)) {
      if (fs.statSync(args.model).isDirectory()) options.modelPath = 'file://' + path.join(args.model, 'model.json');
      else options.modelPath = 'file://' + args.model;
    } else {
      log.error('model not found:', args.model);
      return;
    }
  }
  await tf.enableProdMode();
  await tf.ready();
  await runDetection(args.input, args.output);
}

main();
