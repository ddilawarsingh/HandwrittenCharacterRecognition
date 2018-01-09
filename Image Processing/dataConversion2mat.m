clear;clc;
X=[];
y=[];
for i = 251:350
    for j = 65:90
        temp = strcat(int2str(j),'_',sprintf('%05d',(i)));
        temp = strcat('a_',temp,'.png');
        grayIM = imread(temp);
        binaryIM = imbinarize(grayIM(:,:,1));
        binaryIM = ~ binaryIM;
        featureVec = ones(1,64);
        row = ones(1,8)*8;
        colm = ones(1,8)*8;
        divImage = mat2cell(binaryIM,row,colm);
        for a = 1:64
            zone = divImage{a};
            sum2 = 0;
            for b = 1:8
                k = b;
                m = 1;
                sum1 = 0;
                while k>=1 && m<=b
                    sum1 = sum1 + zone(k,m);
                    k = k - 1;
                    m = m + 1;
                end
                sum2 = sum2 + sum1;
            end
            for b = 7:1
                k = b;
                m = 1;
                sum1 = 0;
                while k>=1 && m<=b
                    sum1 = sum1 + zone(k,m);
                    k = k - 1;
                    m = m + 1;
                end
                sum2 = sum2 + sum1;
            end
            sum2 = sum2/15;
            featureVec(a) = sum2;
        end
        X = [X;featureVec];
        tempY = j-64;
        y = [y;tempY];
    end
end
save('testingDataSet','X','y');