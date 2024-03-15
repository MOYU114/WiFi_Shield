function [] = ampt2fram(timestamp,csi_ampt,name)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明


j = 1;
start = 1;
% flg = 0;
% step = timestamp(end)/0;
% for i = 1:length(timestamp)
%    if timestamp(i) > 0.05*40
%        start = i;
%        break;
%    end
% end
% disp(start)

csi_ampt = wdenoise(hampel(csi_ampt),3);
for i = 1:length(timestamp)
    if j == 1201
        break;
    end
    if timestamp(i) >= 0.05*j && timestamp(i) < 0.05*(j+1)
%          temp = hampel(csi_ampt(start:i-1,:));   % Hampel for CSI frame
%          temp = wdenoise(hampel(csi_ampt(start:i-1,:)));  % Hampel+wdenoise for CSI frame
         temp = csi_ampt(start:i-1,:);
         temp = reshape(temp,[1,size(temp,1)*size(temp,2)]);
         writematrix(temp,strcat(name,'.csv'),'WriteMode','append');
         start = i;
         j = j+1;
    elseif timestamp(i) >= 0.05*j && timestamp(i) > 0.05*(j+1)
        temp = csi_ampt(start,:);
        temp = reshape(temp,[1,size(temp,1)*size(temp,2)]);
        writematrix(temp,strcat(name,'.csv'),'WriteMode','append');
%         temp = 0;
        writematrix(temp,strcat(name,'.csv'),'WriteMode','append');
        start = i;
        j = j+2;
%         flg = 1;
%     elseif timestamp(i) >= 0.05*j && timestamp(i) > 0.05*(j+1) && timestamp(i) > 0.05*(j+2)
%         temp = csi_ampt(start,:);
%         temp = reshape(temp,[1,size(temp,1)*size(temp,2)]);
%         writematrix(temp,strcat(name,'.csv'),'WriteMode','append');
%         temp = 0;
%         writematrix(temp,strcat(name,'.csv'),'WriteMode','append');
%         temp = 0;
%         writematrix(temp,strcat(name,'.csv'),'WriteMode','append');
%         start = i;
%         j = j+3;
%         flg = 1;
    else
        continue;    
    end
%     if i == 2400 && flg == 1
%         disp(name);
%     end
end
end

% for i = 0:49
%     test = [points(1:500,28),csi_in_ave(i+1:i+500,1:25)];
%     test_corr = corr(test);
%     test_max(i) = max(test_corr(1,2:26));
% end