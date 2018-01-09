Disp=[];
for i = 251:350
    for j = 65:90
        temp = strcat(int2str(j),'_',sprintf('%05d',(i)));
        temp = strcat('a_',temp,'.png');
        grayIM = imread(temp);
        binaryIM = imbinarize(grayIM(:,:,1));
        binaryIM = reshape(binaryIM,[1,4096]);
        Disp = [Disp;binaryIM];
    end
end
save('displayDataForTesting','Disp');