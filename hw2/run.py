from data_utils import *
from model_utils import *
from interval_model import *
import gc

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)



#model loader
model = load_model()

#interval model converter
interval_model = IntervalModel(model)

#data loader
train_dataloader, test_dataloader = load_dataloader('mnist_data/')

data = next(iter(test_dataloader))
inputs, labels = data
print("Inputs Shape", inputs.shape) #(B, C, H, W)

# model forward
output = model(inputs)

#create interval
epsilon = 0.1
interval = Interval(inputs-epsilon, inputs+epsilon)
# interval model forward
output_interval = interval_model(interval)
assert output.shape == output_interval.lower.shape and output.shape == output_interval.upper.shape #(B, 10)

output = output.flatten().detach().numpy()
output_interval = output_interval.lower.flatten().detach().numpy(), output_interval.upper.flatten().detach().numpy()

for i in range(len(output)):
    if output[i] < output_interval[0][i] or output[i] > output_interval[1][i]:
        print(f'output: {output[i]}, interval: {output_interval[0][i]} ~ {output_interval[1][i]}')
        raise('error')
    
print('Checking on CUDA')
model = model.cuda()
interval_model = IntervalModel(model).cuda()
inputs = inputs.cuda()
interval = Interval((inputs-epsilon).cuda(), (inputs+epsilon).cuda())
output = model(inputs)
interval_output = interval_model(interval)
assert output.shape == interval_output.lower.shape == interval_output.upper.shape
output = output.flatten().detach().cpu().numpy()
interval_output = interval_output.lower.flatten().detach().cpu().numpy(), interval_output.upper.flatten().detach().cpu().numpy()
for i in range(len(output)):
    if output[i] < interval_output[0][i] or output[i] > interval_output[1][i]:
        print(f'output: {output[i]}, interval: {interval_output[0][i]} ~ {interval_output[1][i]}')
        raise('error')

del model
del interval_model
torch.cuda.empty_cache()
gc.collect()