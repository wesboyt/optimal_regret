import torch

def create_cost_matrix(num_bins, device='cpu'):
    #Generates a normalized 1D squared Euclidean cost matrix.
    x = torch.linspace(0, 1, steps=num_bins, device=device)
    C = (x.unsqueeze(0) - x.unsqueeze(1)) ** 2
    return C

def sinkhorn_loss(p, q, C, epsilon=0.01, num_iters=50):
    #Computes the Sinkhorn divergence (Entropic Optimal Transport loss).
    #Ensure inputs sum to 1 (adding 1e-8 for numerical stability)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
    q = q / (q.sum(dim=-1, keepdim=True) + 1e-8)
    
    K = torch.exp(-C / epsilon)
    u = torch.ones_like(p)
    
    for _ in range(num_iters):
        v = q / (torch.matmul(u, K.t()) + 1e-8)
        u = p / (torch.matmul(v, K) + 1e-8)
        
    K_C = K * C 
    loss = torch.sum(u * torch.matmul(v, K_C.t()), dim=1)
    
    return loss.mean()

def demo_regret_sinkhorn():
    #Setup your specific inputs
    #We wrap them in an extra bracket to simulate a batch size of 1: (1, 3)
    p_raw = torch.tensor([[1.0, 0.0, 2.0]], requires_grad=True) # Predicted: [1, 0, 2]
    q_raw = torch.tensor([[0.0, 3.0, 0.0]])                     # Target: [0, 3, 0]

    #Convert raw sample counts to probability distributions
    #Your model's raw logits would typically pass through a softmax here
    p_prob = p_raw / p_raw.sum(dim=1, keepdim=True)
    q_prob = q_raw / q_raw.sum(dim=1, keepdim=True)

    print("Distributions")
    print(f"Predicted (p): {p_prob[0].tolist()}")
    print(f"Target (q):    {q_prob[0].tolist()}")

    #Create the cost matrix for 3 bins
    num_bins = p_raw.shape[1]
    C = create_cost_matrix(num_bins)
    
    print("Cost Matrix (C)")
    print(C)

    #Calculate the loss
    #Using epsilon=0.05 for smooth convergence in this small demo
    loss = sinkhorn_loss(p_prob, q_prob, C, epsilon=0.05, num_iters=50)
    
    print("Forward Pass")
    print(f"Calculated Sinkhorn Loss: {loss.item():.4f}")

    #Demonstrate backpropagation
    loss.backward()
    
    print("Backward Pass")
    print("Gradients on the raw prediction (p_raw):")
    print(p_raw.grad)

if __name__ == "__main__":
    demo_regret_sinkhorn()
