root='C:\Users\xyz\Downloads\output\1\';
files=dir([root '*.txt']);
outfile=[root 'rst.avi'];
sorted=sort_nat({files.name});

aviobj=VideoWriter(outfile);
aviobj.FrameRate = 2;
open(aviobj)

for i=2:length(sorted)
%     file=files(i).name;
    x=load([root sorted{i}]);
    param=reshape(x,12,12,24,4);
    % param=param.*repmat(param(:,:,:,1)>0.2,1,1,1,4);
    coef=ones(size(param(:,:,:,1)));
    idx=find(param(:,:,:,1)<0.2);
    coef(idx)=NaN;
    param=param.*repmat(coef,1,1,1,4);
    [ny,nz,nx,~]=size(param);
    [X,Y,Z]=meshgrid(1:nx,1:ny,1:nz);
    param=permute(param,[1,3,2,4]);
    h=slice(X,Y,Z,param(:,:,:,1),1:nx,1:ny,1:nz);
    set(h,'FaceColor','interp',...
        'EdgeColor','none')
    camproj perspective
    box on
    view(-30,20)
    axis equal off
    colormap jet
%     colorbar

%     set(gcf, 'visible', 'off');           % 不显示窗口
%     q = get(gca, 'position');  % 获取位置
% %     q(1) = 0;     % 设置左边距离值为零
% %     q(2) = 0;     % 设置右边距离值为零
%     set(gca, 'position', q);
    
    frame = getframe(gcf);
    im = frame2im(frame);      % 制作gif文件，图像必须是index索引图像
    writeVideo(aviobj,im);
%     imshow(im);
%     [I, map] = rgb2ind(im, 256);
%     if i == 2
%         imwrite(I, map, outfile, 'gif', 'Loopcount', inf, 'DelayTime', 0.3);
%     else
%         imwrite(I, map, outfile, 'gif', 'WriteMode', 'append', 'DelayTime', 0.3);
%     end
end
close(aviobj)