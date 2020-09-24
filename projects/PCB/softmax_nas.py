from __future__ import division, print_function, absolute_import

from torchreid import metrics
from torchreid.engine import Engine
from torchreid.losses import CrossEntropyLoss


class ImageSoftmaxNASEngine(Engine):

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=False,
        label_smooth=True,
        ):
        super(ImageSoftmaxNASEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        for k in range(self.mc_iter):
            outputs = self.model(imgs)
            loss = self.compute_loss(self.criterion, outputs, pids)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_dict = {
            'loss': loss.item(),
            'acc': metrics.accuracy(outputs, pids)[0].item()
        }

        return loss_dict
