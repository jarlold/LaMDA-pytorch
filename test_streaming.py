import lamda_pytorch.build_streamable_dataloader

a, b = lamda_pytorch.build_streamable_dataloader.build_dataloaders()

#print(len(a))

c = next(iter(a))

print(c['input_ids'].shape)
print(c['labels'].shape)
print(len(a))
print(len(b))
