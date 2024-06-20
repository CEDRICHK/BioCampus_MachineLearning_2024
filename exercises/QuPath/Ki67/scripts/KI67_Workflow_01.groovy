setImageType('BRIGHTFIELD_H_DAB');
clearSelectedObjects(true);
createFullImageAnnotation(true)
setColorDeconvolutionStains('{"Name" : "H-DAB modified", "Stain 1" : "Hematoxylin", "Values 1" : "0.8017 0.52541 0.28501", "Stain 2" : "DAB", "Values 2" : "0.47506 0.4948 0.72766", "Background" : " 217 219 223"}');
runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '{"detectionImageBrightfield":"Hematoxylin OD","requestedPixelSizeMicrons":0.22,"backgroundRadiusMicrons":3.0, "backgroundByReconstruction":true,"medianRadiusMicrons":0.0,"sigmaMicrons":1.55,"minAreaMicrons":5.0,"maxAreaMicrons":400.0,"threshold":0.078,"maxBackground":2.0,"watershedPostProcess":true,"excludeDAB":false,"cellExpansionMicrons":3.0,"includeNuclei":true,"smoothBoundaries":true,"makeMeasurements":true}')
runObjectClassifier("KI67_ML_01");