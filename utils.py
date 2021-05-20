from copy import deepcopy

def filterDataset(dataset, exclude_classes=None):
    dataset_ = deepcopy(dataset)
    if not exclude_classes:
        exclude_classes = []
    cls_list = dataset_.targets
    data_list = dataset_.data
    exclude_idx = [x for x in range(len(cls_list)) if cls_list[x] in exclude_classes]
    dataset_.targets = [cls_list[x] for x in range(len(cls_list)) if x not in exclude_idx]
    dataset_.data = [data_list[x,:,:] for x in range(len(cls_list)) if x not in exclude_idx]
    return dataset_


