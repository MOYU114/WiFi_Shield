function [] = plotcsi( csi, nfft, normalize )
%PLOTCSI Summary of this function goes here
%   Detailed explanation goes here

csi_buff = fftshift(csi,2);     %左右置换，左边的首和右边的首对换。
csi_phase = rad2deg(angle(csi_buff));   
for cs = 1:size(csi_buff,1)     %csi数据的维度，也就是行数
    csi = abs(csi_buff(cs,:));  %得到csi的幅值
    if normalize
        csi = csi./max(csi);    %归一化操作，每一行最大值为1，其他值为与它的比值。
    end
    csi_buff(cs,:) = csi;
end

figure
x = -(nfft/2):1:(nfft/2-1);
subplot(3,1,3)
imagesc(x,[1 size(csi_buff,1)],csi_buff)
myAxis = axis();
axis([min(x)-0.5, max(x)+0.5, myAxis(3), myAxis(4)])
set(gca,'Ydir','reverse')
xlabel('Subcarrier')
ylabel('Packet number')

max_y = max(csi_buff(:));
for cs = 1:size(csi_buff,1)
    csi = csi_buff(cs,:);
    
    subplot(3,1,1)
    plot(x,csi);
    grid on
    myAxis = axis();
    axis([min(x)-0.5, max(x)+0.5, 0, max_y])
    xlabel('Subcarrier')
    ylabel('Magnitude')
    title('Channel State Information')
    text(max(x),max_y-(0.05*max_y),['Packet #',num2str(cs),' of ',num2str(size(csi_buff,1))],'HorizontalAlignment','right','Color',[0.75 0.75 0.75]);
    
    subplot(3,1,2)
    plot(x,csi_phase(cs,:));
    grid on
    myAxis = axis();
    axis([min(x)-0.5, max(x)+0.5, -180, 180])
    xlabel('Subcarrier')
    ylabel('Phase')
    %disp('Press any key to continue..');
    %waitforbuttonpress();
end
%close

end

