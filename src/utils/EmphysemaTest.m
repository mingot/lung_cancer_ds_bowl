clear;
close all;


voxel_size = 0.78 * 0.78 * 1.25;


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
FeaturesPatch = zeros(nPatches,4);
GroundTruthPatch = zeros(nPatches,1);
for i = 1:nPatches
    label = patch_labels(i);
    image_filename = [DatabasePath,'patches',filesep,'patch',sprintf('%d',i),'.tiff'];
    im = double(imread(image_filename));

    % Compute features 1 and 2
    gated_image = im(im<Threshold);
    min_value = min(gated_image(:));
    max_value = max(gated_image(:));
    edges = min_value:max_value;
    pixelCounts = histcounts(gated_image,edges);
    GLs = edges(2:end);
    [skew, kurtosis] = GetSkewAndKurtosis(GLs, pixelCounts);
    FeaturesPatch(i,1) = skew;
    FeaturesPatch(i,2) = kurtosis;

    % Compute features 3 and 4
    % Mean Lung Density Method
    mld = mean(im(:))/voxel_size;
    FeaturesPatch(i,3) = mld;
    % Histogram Analysis
    min_value = min(im(:));
    max_value = max(im(:));
    edges = min_value:max_value;
    pixelCounts = histcounts(im,edges);
    GLs = edges(2:end)';
    accumulated = 100*cumsum(pixelCounts)/sum(pixelCounts);
    index = find(accumulated>20);
    fifth_percentile = GLs(index(1));
    FeaturesPatch(i,4) = fifth_percentile;
    
    if label==1
        GroundTruthPatch(i) = 0;
    else
        GroundTruthPatch(i) = 1;
    end
end

NonEmphysemaPatch = find(GroundTruthPatch==0);
EmphysemaPatch = find(GroundTruthPatch==1);


csvwrite([DatabasePath,'results.csv'],[FeaturesPatch(:,3:4),GroundTruthPatch]);

%%

% fileID = fopen([DatabasePath,'slice_labels.csv']);
% C = textscan(fileID,'%s%d','Delimiter',',');
% slices_labels_text = C{1,1};
% slices_labels = C{1,2};
% fclose(fileID);
% 
% nSlices = length(slices_labels);
% FeaturesSlice = zeros(nSlices,4);
% GroundTruthSlice = zeros(nSlices,1);
% for i = 1:nSlices
%     label = slices_labels(i);
%     image_filename = [DatabasePath,'slices',filesep,slices_labels_text{i},'.tiff'];
%     im = double(imread(image_filename));
%     % Compute features 1 and 2
%     gated_image = im(im<Threshold);
%     min_value = min(gated_image(:));
%     max_value = max(gated_image(:));
%     edges = min_value:max_value;
%     pixelCounts = histcounts(gated_image,edges);
%     GLs = edges(2:end);
%     [skew, kurtosis] = GetSkewAndKurtosis(GLs, pixelCounts);
%     FeaturesSlice(i,1) = skew;
%     FeaturesSlice(i,2) = kurtosis;
% 
%     % Compute features 3 and 4
%     % Mean Lung Density Method
%     
%     figure;
%     imagesc(im);
%     
%     mld = mean(im(:))/voxel_size;
%     FeaturesSlice(i,3) = mld;
%     % Histogram Analysis
%     min_value = min(im(:));
%     max_value = max(im(:));
%     edges = min_value:max_value;
%     pixelCounts = histcounts(im,edges);
%     GLs = edges(2:end)';
%     accumulated = 100*cumsum(pixelCounts)/sum(pixelCounts);
%     index = find(accumulated>20);
%     fifth_percentile = GLs(index(1));
%     FeaturesSlice(i,4) = fifth_percentile;
%     
%     figure;
%     subplot(2,1,1);
%     plot(GLs,pixelCounts);
%     subplot(2,1,2);
%     plot(GLs,accumulated);
%     hold on;
%     scatter(fifth_percentile,20);
%     
%     if label==1
%         GroundTruthSlice(i) = 0;
%     else
%         GroundTruthSlice(i) = 1;
%     end
% end
% 
% NonEmphysemaSlice = find(GroundTruthSlice==0);
% EmphysemaSlice = find(GroundTruthSlice==1);
% 
% figure;
% subplot(2,1,1)
% scatter(FeaturesPatch(EmphysemaPatch,3),FeaturesPatch(EmphysemaPatch,4));
% hold on;
% grid on;
% scatter(FeaturesPatch(NonEmphysemaPatch,3),FeaturesPatch(NonEmphysemaPatch,4));
% legend('Emphysema','NonEmphysema');
% xlabel('mld');
% ylabel('fifth percentile');
% title('Patches');
% 
% subplot(2,1,2)
% scatter(FeaturesSlice(EmphysemaSlice,3),FeaturesSlice(EmphysemaSlice,4));
% hold on;
% grid on;
% scatter(FeaturesSlice(NonEmphysemaSlice,3),FeaturesSlice(NonEmphysemaSlice,4));
% legend('Emphysema','NonEmphysema');
% xlabel('mld');
% ylabel('fifth percentile');
% title('Patches');
% 
% 
% 
% %%
% GroundTruth = [GroundTruthPatch;GroundTruthSlice];
% Features = [FeaturesPatch;FeaturesSlice];
% 
% NonEmphysema = find(GroundTruth==0);
% Emphysema = find(GroundTruth==1);
% 
% %%
% 
% figure;
% scatter(Features(Emphysema,1),Features(Emphysema,2));
% hold on;
% scatter(Features(NonEmphysema,1),Features(NonEmphysema,2));
% legend('Emphysema','NonEmphysema');
% 
% 
% %%
% csvwrite([DatabasePath,'results.csv'],[Features,GroundTruth]);
% 


    