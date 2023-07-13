const fs = require('fs');
const { ExifTool } = require("exiftool-vendored");
const path = require('path');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');
const argparse = require('argparse');

const { Canvas, loadImage } = require('canvas'); // eslint-disable-line node/no-unpublished-require

const options = { // options
  debug: true,
  modelPath: 'file://models/default-f16/model.json',
  minScore: 0.38,
  maxResults: 50,
  iouThreshold: 0.5,
  outputNodes: ['output1', 'output2', 'output3'],
  blurNude: false,
  blurRadius: 25,
  labels: undefined, // can be base or default
  classes: { // classes labels
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
  composite: undefined, // can be base or default
  composites: { // composite definitions of what is a person, sexy, nude
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

const models = []; // holds instance of graph model

// draw rect with rounded corners
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

// read image file and prepare tensor for further processing
function getTensorFromImage(imageFile) {
  if (!fs.existsSync(imageFile)) {
    log.error('Not found:', imageFile);
    return null;
  }
  const data = fs.readFileSync(imageFile);
  const bufferT = tf.node.decodeImage(data, 3);
  const expandedT = tf.expandDims(bufferT, 0);
  const imageT = tf.cast(expandedT, 'float32');
  imageT['file'] = imageFile;
  tf.dispose([expandedT, bufferT]);
  if (options.debug) log.info('loaded image:', imageT['file'], 'width:', imageT.shape[2], 'height:', imageT.shape[1]);
  return imageT;
}

// create output jpeg after processing
async function saveProcessedImage(inImage, outImage, data) {
  if (!data) return false;
  return new Promise(async (resolve) => { // eslint-disable-line no-async-promise-executor
    const original = await loadImage(inImage); // load original image
    const c = new Canvas(original.width, original.height); // create canvas
    const ctx = c.getContext('2d');
    ctx.drawImage(original, 0, 0, c.width, c.height); // draw original onto output canvas
    for (const obj of data.parts) { // draw all detected objects
      if (options.composite.nude.includes(obj.id) && options.blurNude) blur({ canvas: c, left: obj.box[0], top: obj.box[1], width: obj.box[2], height: obj.box[3] });
      rect({ canvas: c, x: obj.box[0], y: obj.box[1], width: obj.box[2], height: obj.box[3], title: `${Math.round(100 * obj.score)}% ${obj.class}` });
    }
     const tag = (data.nude === true || data.nude === 'true') ? 'NSFW' : 'SFW';
    const out = fs.createWriteStream(outImage); // write canvas to jpeg
    out.on('finish', () => {
      if (options.debug) log.state('created output image:', outImage);
      resolve(true);
    });
    out.on('error', (err) => {
      log.error('error creating image:', outImage, err);
      resolve(true);
    });
    const stream = c.createJPEGStream({ quality: 0.6, progressive: true, chromaSubsampling: true });
    stream.pipe(out);
  });
    const tag = (data.nude === true || data.nude === 'true') ? 'NSFW' : 'SFW';
      await addMetadata(inImage, tag);
         if (options.debug) log.state('Added metadata to:', inImage);

}

// parse prediction data
async function processPrediction(boxesTensor, scoresTensor, classesTensor, inputTensor) {
  const boxes = await boxesTensor.array();
  const scores = await scoresTensor.data();
  const classes = await classesTensor.data();
  const nmsT = await tf.image.nonMaxSuppressionAsync(boxes[0], scores, options.maxResults, options.iouThreshold, options.minScore); // sort & filter results
  const nms = await nmsT.data();
  tf.dispose(nmsT);
  const parts = [];
  for (const i in nms) { // create body parts object
    const id = parseInt(i);
    parts.push({
      score: scores[i],
      id: classes[id],
      class: options.labels[classes[id]], // lookup classes
      box: [ // convert box from x0,y0,x1,y1 to x,y,width,heigh
        Math.trunc(boxes[0][id][0]),
        Math.trunc(boxes[0][id][1]),
        Math.trunc((boxes[0][id][3] - boxes[0][id][1])),
        Math.trunc((boxes[0][id][2] - boxes[0][id][0])),
      ],
    });
  }
  const result = {
    input: { file: inputTensor.file, width: inputTensor.shape[2], height: inputTensor.shape[1] },
    person: parts.filter((a) => options.composite.person.includes(a.id)).length > 0,
    sexy: parts.filter((a) => options.composite.sexy.includes(a.id)).length > 0,
    nude: parts.filter((a) => options.composite.nude.includes(a.id)).length > 0,
    parts,
  };
  if (options.debug) log.data('result:', result);
  return result;
}

// load graph model and run inference
async function runDetection(input, output) {
  const t = {};
  if (!models[options.modelPath]) { // load model if not already loaded
    try {
      models[options.modelPath] = await tf.loadGraphModel(options.modelPath);
      models[options.modelPath].path = options.modelPath;
      if (options.debug) log.state('loaded graph model:', options.modelPath);
      if (models[options.modelPath].version === 'v2.base') {
        options.labels = options.classes.base;
        options.composite = options.composites.base;
      } else {
        options.labels = options.classes.default;
        options.composite = options.composites.default;
      }
    } catch (err) {
      log.error('error loading graph model:', options.modelPath, err.message, err);
      return null;
    }
  }
  t.input = getTensorFromImage(input); // get tensor from image
  [t.boxes, t.scores, t.classes] = await models[options.modelPath].executeAsync(t.input, options.outputNodes); // run prediction
  const res = await processPrediction(t.boxes, t.scores, t.classes, t.input); // parse outputs
  if (output) await saveProcessedImage(input, output, res); // save processed image and return result
  Object.keys(t).forEach((tensor) => tf.dispose(t[tensor])); // free up memory
  return res;
}
// Add metadata to PNG
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


// main function
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
