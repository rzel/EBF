function ShowMatches_color_split(I1,I2,x1,y1,x2,y2, linewidth, color, show_im, left, filename)

figure
if (~exist('linewidth', 'var'))
    linewidth=3;
end;

if size(I1,1)> size(I2,1)
    I2(size(I1,1), size(I1,2), size(I1,3))=0;
else
    I1(size(I2,1), size(I2,2), size(I2,3))=0;
end;


if left
    
    
    if show_im
        handle_a=imshow(I1);
    end;
    hold on;
    plot(x1,y1,'.','markersize',10*linewidth,'Color',color);
    hl=line([x1,x2]',[y1,y2]');
    set(hl,'LineWidth',linewidth,'Color',color)
    
    
    if exist('filename', 'var')
        saveas(handle_a,filename);
    end;
    
    return;
end;
% quiver(x1,y1,x2-x1,y2-y1, 0)

if show_im
    handle_a=imshow(I2);
    hold on;
end;
plot(x2,y2,'.','markersize',10*linewidth,'Color', color);


if exist('filename', 'var')
    saveas(handle_a,filename);
end;


