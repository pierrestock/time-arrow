require 'image'

local path = '/workdir/stockp/object/youtube-reduce'
-- get names of folders
handle = assert(io.popen('ls -1v ' .. path))
folderNames = string.split(assert(handle:read('*a')), '\n')

-- scale all the images
for i = 1, #folderNames do
    folder = folderNames[i]
    print('Treating folder:', folder)
    if string.sub(folder, 1, 1) == 'F' then
        handle = assert(io.popen('ls ' .. path .. '/' .. folder))
    else
        handle = assert(io.popen('ls ' .. path .. '/' .. folder .. '| sort -r'))
    end
    imageNames = string.split(assert(handle:read('*a')), '\n')
    for j = 1, #imageNames do
        imagePath = path .. '/' .. folder .. '/' .. imageNames[j]
        img = image.load(imagePath)
        img = image.scale(img, 256, 256)
        image.save(imagePath, img)
    end
end

print('All images rescaled !')
