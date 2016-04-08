function [f1, f2, d1, d2] = ASIFT(img1, img2, numTilts, resize)

 
    [f1, f2, d1, d2] = mexASIFT(img1', img2', numTilts, resize);


end