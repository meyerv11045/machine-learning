import torch

def train_model(model, criterion, optimizer, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            y_hat = model(x.float())
            loss = criterion(y_hat, y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                print(f'epoch {epoch + 1} / {epochs}, loss =  {loss.item():.4f}')
    
    return model

def test_model(model,criterion,test_loader):
    """ Evaluate trained model on the testing data
        using the specified criterion/loss function 
    """
    model.eval()
    with torch.no_grad():
        loss = 0
        n_samples = len(test_loader)
        for (x,y) in test_loader:
            y_hat = model(x.float())
            loss += criterion(y_hat,y.float()).item()
        
        print(f'Total Test Loss = {loss:.4f}')
        print(f'Avg Loss per Sample = {loss/n_samples:.4f} ')