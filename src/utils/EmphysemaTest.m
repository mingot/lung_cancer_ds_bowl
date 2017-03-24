clear;
close all;

% Patches labels
%  1 normal tissue, 2 centrilobular emphysema, 3 paraseptal emphysema.

% Slices labels
%  1 normal tissue, 2 centrilobular emphysema, 3 paraseptal emphysema, 4 panlobular emphysema

Threshold = -600;

DatabasePath = '/Users/rdg/Documents/MATLAB/emphysema/';

fileID = fopen([DatabasePath,'patch_labels.csv']);
C = textscan(fileID,'%s%d','Delimiter',',');
patch_labels_text = C{1,1};
patch_labels = C{1,2};
fclose(fileID);

nPatches = length(patch_labels);
FeaturesPatch = zeros(nPatches,2);
GroundTruthPatch = zeros(nPatches,1);
for i = 1:nPatches
    label = patch_labels(i);
    image_filename = [DatabasePath,'patches',filesep,'patch',sprintf('%d',i),'.tiff'];
    image = double(imread(image_filename));
    edges = -1000:1000;
    gated_image = image(image<Threshold);
        
    h = histogram(gated_image,edges);
    GLs = edges(2:end)';
    pixelCounts = h.Values';
    [skew, kurtosis] = GetSkewAndKurtosis(GLs, pixelCounts);
    FeaturesPatch(i,1) = skew;
    FeaturesPatch(i,2) = kurtosis;
    if label==1
        GroundTruthPatch(i) = 0;
    else
        GroundTruthPatch(i) = 1;
    end
end


NonEmphysema = find(GroundTruthPatch==0);
Emphysema = find(GroundTruthPatch==1);

%%

h = histogram(gated_image,edges);

%%
figure;
scatter(FeaturesPatch(Emphysema,1),FeaturesPatch(Emphysema,2));
hold on;
scatter(FeaturesPatch(NonEmphysema,1),FeaturesPatch(NonEmphysema,2));
legend('Emphysema','NonEmphysema');
xlabel('skew');
ylabel('kurtosis');

%

fileID = fopen([DatabasePath,'slice_labels.csv']);
C = textscan(fileID,'%s%d','Delimiter',',');
slices_labels_text = C{1,1};
slices_labels = C{1,2};
fclose(fileID);

nSlices = length(slices_labels);
FeaturesSlice = zeros(nSlices,2);
GroundTruthSlice = zeros(nSlices,1);
for i = 1:nSlices
    label = slices_labels(i);
    image_filename = [DatabasePath,'slices',filesep,slices_labels_text{i},'.tiff'];
    image = double(imread(image_filename));
    edges = -1000:1000;
    gated_image = image(image<Threshold);
    h = histogram(gated_image,edges);
    GLs = edges(2:end)';
    pixelCounts = h.Values';
    [skew, kurtosis] = GetSkewAndKurtosis(GLs, pixelCounts);
    FeaturesSlice(i,1) = skew;
    FeaturesSlice(i,2) = kurtosis;
    if label==1
        GroundTruthSlice(i) = 0;
    else
        GroundTruthSlice(i) = 1;
    end
end

%%
GroundTruth = [GroundTruthPatch;GroundTruthSlice];
Features = [FeaturesPatch;FeaturesSlice];

NonEmphysema = find(GroundTruth==0);
Emphysema = find(GroundTruth==1);

%%

figure;
scatter(Features(Emphysema,1),Features(Emphysema,2));
hold on;
scatter(Features(NonEmphysema,1),Features(NonEmphysema,2));
legend('Emphysema','NonEmphysema');


%%
csvwrite([DatabasePath,'results.csv'],[Features,GroundTruth]);



    