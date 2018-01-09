X=[];
y=[];
for i = 1:25
    for j = 97:122
        temp = strcat(char(j),int2str(i));
        temp = strcat(temp,'.bmp');
        rgbIm = imread(temp);
        grayIM = rgb2gray(rgbIm);
        binaryIM = imbinarize(grayIM);
        binaryIM = reshape(binaryIM,[1,2500]);
        X = [X;binaryIM];
        tempY = j-96;
        y = [y;tempY];
    end
end
save('dataSet','X','y');