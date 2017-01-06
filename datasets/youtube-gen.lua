local M = {}

local function findImages(path)

    imagePaths = {} -- path to the images
    videoNames = {} -- name of the video the image belongs to

    -- get names of folders
    handle = assert(io.popen('ls -1v ' .. path))
    folderNames = string.split(assert(handle:read('*a')), '\n')

    -- get all the images
    for i = 1, #folderNames do
        folder = folderNames[i]
        if string.sub(folder, 1, 1) == 'F' then
            handle = assert(io.popen('ls ' .. path .. '/' .. folder))
        else
            handle = assert(io.popen('ls ' .. path .. '/' .. folder .. '| sort -r'))
        end
        imageNames = string.split(assert(handle:read('*a')), '\n')
        for j = 1, #imageNames do
            image = imageNames[j]
            table.insert(imagePaths, image)
            table.insert(videoNames, folder)
        end
    end
    return imagePaths, videoNames
end


function M.exec(opt, cacheFile)

   local trainPath = opt.data .. '/train'
   local valPath = opt.data .. '/val'

   print("finding all validation images...")
   local trainImageNames, trainVideoNames = findImages(trainPath)

   print("finding all training images...")
   local valImageNames, valVideoNames = findImages(valPath)

   local info = {
      path = opt.data,
      train = {
         imageNames = trainImageNames,
         videoNames = trainVideoNames,
      },
      val = {
         imageNames = valImageNames,
         videoNames = valVideoNames,
      },
   }

   print("saving list of images to " .. cacheFile .. "...")
   torch.save(cacheFile, info)

   return info
end

return M
