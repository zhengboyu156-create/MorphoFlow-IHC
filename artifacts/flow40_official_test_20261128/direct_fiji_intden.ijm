args = split(getArgument(), ",");
if (args.length != 2) exit("Expected input_dir,output_csv");
inputDir = args[0];
if (!(endsWith(inputDir, "\\")) && !(endsWith(inputDir, "/"))) inputDir = inputDir + "\\";
outCsv = args[1];
list = getFileList(inputDir);
File.saveString("filename,width,height,bit_depth,area,mean,IntDen\n", outCsv);
setBatchMode(true);
count = 0;
for (i=0; i<list.length; i++) {
    name = list[i];
    lower = toLowerCase(name);
    if (!(endsWith(lower, ".png") || endsWith(lower, ".jpg") || endsWith(lower, ".jpeg") || endsWith(lower, ".tif") || endsWith(lower, ".tiff"))) continue;
    open(inputDir + name);
    run("Colour Deconvolution", "vectors=[H DAB]");
    titles = getList("image.titles");
    target = "";
    for (j=0; j<titles.length; j++) if (indexOf(titles[j], "Colour_2") >= 0) target = titles[j];
    if (target == "") { print("ERROR: Colour_2 missing for " + name); run("Close All"); exit("Colour_2 missing"); }
    selectWindow(target);
    width = getWidth(); height = getHeight();
    run("8-bit"); depth = bitDepth();
    run("Set Measurements...", "area mean integrated redirect=None decimal=6");
    run("Measure"); row = nResults - 1;
    area = getResult("Area", row); mean = getResult("Mean", row); intden = getResult("IntDen", row);
    File.append(name + "," + width + "," + height + "," + depth + "," + area + "," + mean + "," + intden + "\n", outCsv);
    run("Clear Results"); run("Close All");
    count++;
    if (count % 100 == 0) print("measured " + count + " images");
}
setBatchMode(false);
File.saveString(count + "\n", outCsv + ".done");
print("DONE count=" + count + " csv=" + outCsv);
run("Quit");
