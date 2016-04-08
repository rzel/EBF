function [f1, f2, d1, d2] = ASIFT(img1, img2, numTilts, resize)
    [f1,d1] = mexASIFT(img1', numTilts, resize);
    [f2,d2] = mexASIFT(img2', numTilts, resize);
end