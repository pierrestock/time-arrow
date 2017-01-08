rootPath = '/Users/pstock/Desktop/Object Recognition/Project/Code/youtube-reduce/train';
infos = dir(rootPath);
folderNames = {infos.name};
folderNames = folderNames(4:end);

norms = [];

for i = 1:length(folderNames)
    disp(folderNames(i))
    opticFlow = opticalFlowFarneback;
    folderPath = strcat(rootPath, '/', folderNames(i));
    infos = dir(folderPath{1});
    fileNames = {infos.name};
    fileNames = fileNames(3:end);
    if strcmp(fileNames(1), '.DS_Store')
        fileNames = fileNames(2:end);
    end
    for j = 1:length(fileNames)
        imagePath = strcat(folderPath, '/', fileNames(j));
        img = imread(imagePath{1});
        img = rgb2gray(img);
        flow = estimateFlow(opticFlow, img);
        norms(i, j) = mean(flow.Magnitude(:));
    end
end

% plot average flow for all videos
plot(sort(mean(norms,2), 'descend'), 'linewidth', 2);
xlabel('Video indexes');
xlim([0 size(norms,1)]);
ylabel('Average optical flow magnitude');
set(gca,'fontsize', 35);

% get video indexes
[~, idx_m] = sort(mean(norms,2), 'descend');

% action videos
for i = 1:size(norms,1)/2
    disp(cell2mat(folderNames(idx_m(i))))
end

% rest videos
for i = (size(norms,1)/2 + 1):size(norms,1)
    disp(cell2mat(folderNames(idx_m(i))))
end

% compare first and last videos 
plot(norms(idx_m(1),:), 'linewidth', 2); hold on;
plot(norms(idx_m(end),:), 'linewidth', 2);
xlabel('Frame indexes');
legend('Most intense video', 'Least intense video')
xlim([0 240]);
ylabel('Optical flow magnitude');
set(gca,'fontsize', 35);
