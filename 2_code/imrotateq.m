function im = imrotateq(im, ang)
    im = imcomplement(im);
    im = imrotate(im, ang, 'crop');
    im = imcomplement(im);
end