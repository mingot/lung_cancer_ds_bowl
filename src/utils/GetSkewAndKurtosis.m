%------------------------------------------------------------------------------------------------------
% Get the skew and kurtosis from the histogram bin values.
% Uses formulas from http://itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
function [skew, kurtosis] = GetSkewAndKurtosis(GLs, pixelCounts)
	try
		% Get the number of pixels in the histogram.
		numberOfPixels = sum(pixelCounts);
		% Get the mean gray lavel.
		meanGL = sum(GLs .* pixelCounts) / numberOfPixels;
		% Get the variance, which is the second central moment.
		varianceGL = sum((GLs - meanGL) .^ 2 .* pixelCounts) / (numberOfPixels-1);
		% Get the standard deviation.
		sd = sqrt(varianceGL);
		% Get the skew.
		skew = sum((GLs - meanGL) .^ 3 .* pixelCounts) / ((numberOfPixels - 1) * sd^3);
		% Get the kurtosis.
		kurtosis = sum((GLs - meanGL) .^ 4 .* pixelCounts) / ((numberOfPixels - 1) * sd^4);
	catch ME
		errorMessage = sprintf('Error in GetSkewAndKurtosis().\nThe error reported by MATLAB is:\n\n%s', ME.message);
		uiwait(warndlg(errorMessage));
		set(handles.txtInfo, 'String', errorMessage);
	end
	return; % from GetSkewAndKurtosis