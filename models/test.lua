-- the entire model will be made of 3 parts {shared, concat, classifier}
net = nn.Sequential()

-- shared part + concatenate part
shared = nn.Parallel(1, 1)

--model = require 'alexnet'
reshape = nn.View(4096 ,1)
model:add(reshape)
fb1 = model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
fb2 = model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
fb3 = model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')

shared:add(fb1)
shared:add(fb2)
shared:add(fb3)

-- classifier part
classifier = nn.Sequential()
reshape = nn.View(3 * 4096)
classifier:add(reshape)
classifier:add(nn.Linear(3 * 4096, 1))

-- build entire model
net:add(shared)
net:add(classifier)

-------------------------------------
t=nn.Sequential()
t:add(nn.Linear(3,2))
t:add(nn.Reshape(1,2))

t1 = t:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
t2 = t:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
t3 = t:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')

c:add(t1)
c:add(t2)
c:add(t3)

classifier = nn.Sequential()
classifier:add(Linear(6,1))

mlp:add(c)
mlp:add(classifier)

x = torch.randn(3,3);
y = torch.ones(1,6);
pred = mlp:forward(x);

criterion = nn.MSECriterion()
err = criterion:forward(pred,y)
gradCriterion = criterion:backward(pred,y);
mlp:zeroGradParameters();
mlp:backward(x, gradCriterion);
mlp:updateParameters(0.01);
