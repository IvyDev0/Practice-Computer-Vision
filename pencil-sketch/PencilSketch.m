function I = PencilSketch(im, stL, width, dirNum, gammaS, gammaI)
%
%   Paras:
%   @im        : the input image.
%   @stL        : the ratio of the length of convolution line to the width of the image.
%   @width     : the width of the stroke.
%   @dirNum    : the number of directions.
%   @gammaS    : the darkness of the stroke.
%   @gammaI    : the darkness of the resulted image.

    %% Read the image
    im = im2double(im);
    [H, W, ~] = size(im);
    if nargin == 1
        ks = ceil(W*stL);
        width = 1;
        dirNum = 8;
        gammaS = 1.0;
        gammaI = 1.0;
    else
        ks = ceil(W/30);
    end

    %% Convert from RGB to YUV
    yuvIm = rgb2ycbcr(im);
    lumIm = yuvIm(:,:,1);

    %% Generate the stroke map
    S = GenStroke(lumIm, ks, width, dirNum) .^ gammaS; % darken the result by gamma
    figure, imshow(S)

    %% Generate the tone map
    J = GenToneMap(lumIm) .^ gammaI; % darken the result by gamma
    figure, imshow(J)

    %% Read the pencil texture
    P = im2double(imread('pencils/pencil0.jpg'));
    P = rgb2gray(P);

    %% Generate the pencil map
    T = GenPencil(lumIm, P, J);
%     figure, imshow(T)

    %% Compute the result
    lumIm = S .* T;

%     if (sc == 3)
%         yuvIm(:,:,1) = lumIm;
%         resultIm = lumIm;
%         I = ycbcr2rgb(yuvIm);
%     else
         I = lumIm;
%     end
end

