import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# Assuming these functions are already defined in utils.model_utils
from utils import equivariant_layer, invariant_layer, sample_dosages, sample_X, sample_Z

torch.autograd.set_detect_anomaly(True)

class SCIGAN_Model:
    def __init__(self, params):
        self.num_features = params['num_features']
        self.num_treatments = params['num_treatments']
        self.export_dir = params['export_dir']

        self.h_dim = params['h_dim']
        self.h_inv_eqv_dim = params['h_inv_eqv_dim']
        self.batch_size = params['batch_size']
        self.alpha = params['alpha']
        self.num_dosage_samples = params['num_dosage_samples']

        self.size_z = self.num_treatments * self.num_dosage_samples
        self.num_outcomes = self.num_treatments * self.num_dosage_samples

        # Set random seed for reproducibility
        torch.manual_seed(10)
        
        # Define device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model components
        self._build_generator()
        self._build_dosage_discriminator()
        self._build_treatment_discriminator()
        self._build_inference_network()
        
        # Initialize optimizers
        self.G_optimizer = optim.Adam(self.generator_params, lr=0.001)
        self.D_dosage_optimizer = optim.Adam(self.dosage_discriminator_params, lr=0.001)
        self.D_treatment_optimizer = optim.Adam(self.treatment_discriminator_params, lr=0.001)
        self.I_optimizer = optim.Adam(self.inference_params, lr=0.001)

    def _build_generator(self):
        # Shared layer
        self.G_shared = nn.Sequential(
            nn.Linear(self.num_features + self.num_treatments + self.size_z, self.h_dim),
            nn.ELU()
        )
        
        # Treatment specific layers
        self.G_treatment_layers = nn.ModuleList()
        for t in range(self.num_treatments):
            treatment_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.h_dim + 1, self.h_dim),
                    nn.ELU()
                ),
                nn.Sequential(
                    nn.Linear(self.h_dim, self.h_dim),
                    nn.ELU()
                ),
                nn.Linear(self.h_dim, 1)
            ])
            self.G_treatment_layers.append(treatment_layers)
            
        self.generator_params = list(self.G_shared.parameters())
        for treatment_layer in self.G_treatment_layers:
            for layer in treatment_layer:
                self.generator_params.extend(list(layer.parameters()))

    def _build_dosage_discriminator(self):
        # Patient features representation
        self.D_dosage_features = nn.Sequential(
            nn.Linear(self.num_features, self.h_dim),
            nn.ELU()
        )
        
        # Treatment specific output layers
        self.D_dosage_outputs = nn.ModuleList()
        for t in range(self.num_treatments):
            self.D_dosage_outputs.append(nn.Linear(self.h_inv_eqv_dim, 1))
            
        self.dosage_discriminator_params = list(self.D_dosage_features.parameters())
        for output_layer in self.D_dosage_outputs:
            self.dosage_discriminator_params.extend(list(output_layer.parameters()))

    def _build_treatment_discriminator(self):
        # Patient features representation
        self.D_treatment_features = nn.Sequential(
            nn.Linear(self.num_features, self.h_dim),
            nn.ELU()
        )
        
        # Shared representation layer
        self.D_treatment_rep = nn.Sequential(
            nn.Linear(self.num_treatments * self.h_inv_eqv_dim + self.h_dim, self.h_dim),
            nn.ELU()
        )
        
        # Output layer
        self.D_treatment_output = nn.Linear(self.h_dim, self.num_treatments)
        
        self.treatment_discriminator_params = list(self.D_treatment_features.parameters())
        self.treatment_discriminator_params.extend(list(self.D_treatment_rep.parameters()))
        self.treatment_discriminator_params.extend(list(self.D_treatment_output.parameters()))

    def _build_inference_network(self):
        # Shared layer
        self.I_shared = nn.Sequential(
            nn.Linear(self.num_features, self.h_dim),
            nn.ELU()
        )
        
        # Treatment specific layers
        self.I_treatment_layers = nn.ModuleList()
        for t in range(self.num_treatments):
            treatment_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.h_dim + 1, self.h_dim),
                    nn.ELU()
                ),
                nn.Sequential(
                    nn.Linear(self.h_dim, self.h_dim),
                    nn.ELU()
                ),
                nn.Linear(self.h_dim, 1)
            ])
            self.I_treatment_layers.append(treatment_layers)
            
        self.inference_params = list(self.I_shared.parameters())
        for treatment_layer in self.I_treatment_layers:
            for layer in treatment_layer:
                self.inference_params.extend(list(layer.parameters()))

    def generator(self, x, y, t, d, z, treatment_dosage_samples):
        # Move inputs to device
        x = x.to(self.device)
        y = y.to(self.device)
        t = t.to(self.device)
        d = d.to(self.device)
        z = z.to(self.device)
        d = d.unsqueeze(1)
        t = t.unsqueeze(1)

        
        treatment_dosage_samples = treatment_dosage_samples.to(self.device)
        
        # Concatenate inputs
        inputs = torch.cat([x, y, t, d, z], dim=1)
        G_shared_out = self.G_shared(inputs)
        
        G_treatment_dosage_outcomes = {}
        
        for treatment in range(self.num_treatments):
            treatment_dosages = treatment_dosage_samples[:, treatment]
            treatment_dosages = treatment_dosages.reshape(-1, 1)
            

            # Expand shared representation for all dosage samples
            G_shared_expand = G_shared_out.repeat(1, self.num_dosage_samples).reshape(-1, self.h_dim)
            
            # Concatenate with treatment dosages
            input_counterfactual_dosage = torch.cat([G_shared_expand, treatment_dosages], dim=1)
            
            # Forward through treatment-specific layers
            treatment_layer_1_out = self.G_treatment_layers[treatment][0](input_counterfactual_dosage)
            treatment_layer_2_out = self.G_treatment_layers[treatment][1](treatment_layer_1_out)
            #treatment_dosage_output = self.G_treatment_layers[treatment][2](treatment_layer_2_out)
            # Change this line in the generator method:
            treatment_dosage_output = self.G_treatment_layers[treatment][2](treatment_layer_2_out)
            
            # Reshape output
            dosage_counterfactuals = treatment_dosage_output.reshape(-1, self.num_dosage_samples)
           
            G_treatment_dosage_outcomes[treatment] = dosage_counterfactuals
            
            
        # Combine outcomes from all treatments
        G_logits = torch.cat([G_treatment_dosage_outcomes[t] for t in range(self.num_treatments)], dim=1)
        G_logits = G_logits.reshape(-1, self.num_treatments, self.num_dosage_samples)
        
        return G_logits, G_treatment_dosage_outcomes

    def dosage_discriminator(self, x, y, treatment_dosage_samples, treatment_dosage_mask, G_treatment_dosage_outcomes):
        # Move inputs to device
        x = x.to(self.device)
        y = y.to(self.device)
        treatment_dosage_samples = treatment_dosage_samples.to(self.device)
        treatment_dosage_mask = treatment_dosage_mask.to(self.device)
        
        # Get patient features representation
        patient_features_representation = self.D_dosage_features(x).unsqueeze(1)  # Add dimension for broadcasting
        
        D_dosage_outcomes = {}
        for treatment in range(self.num_treatments):
            treatment_mask = treatment_dosage_mask[:, treatment]
            treatment_dosages = treatment_dosage_samples[:, treatment]
            
            # Combine factual and generated outcomes
            factual_outcomes = treatment_mask * y
            
            counterfactual_outcomes = (1 - treatment_mask) * G_treatment_dosage_outcomes[treatment]
            treatment_outcomes = factual_outcomes + counterfactual_outcomes
            
            # Prepare inputs for equivariant layer
            dosage_samples = treatment_dosages.unsqueeze(-1)  # [batch, num_dosage_samples, 1]
            dosage_potential_outcomes = treatment_outcomes.unsqueeze(-1)  # [batch, num_dosage_samples, 1]
            
            # Concatenate along last dimension
            inputs = torch.cat([dosage_samples, dosage_potential_outcomes], dim=-1)  # [batch, num_dosage_samples, 2]
            
            # Instanciation des couches équivariantes
            equivariant_layer_1 = equivariant_layer(inputs_dim=inputs.shape[-1], h_dim=self.h_inv_eqv_dim, layer_id=1, treatment_id=treatment)
            equivariant_layer_2 = equivariant_layer(inputs_dim=self.h_inv_eqv_dim, h_dim=self.h_inv_eqv_dim, layer_id=2, treatment_id=treatment)

            # Application des couches équivariantes
            D_h1 = torch.nn.functional.elu(
                equivariant_layer_1(inputs) + patient_features_representation  # Appel de l'instance avec les entrées
            )
            D_h2 = torch.nn.functional.elu(
                equivariant_layer_2(D_h1)  # Appel de l'instance avec D_h1
            )
            
            # Apply treatment-specific output layer
            D_logits_treatment = self.D_dosage_outputs[treatment](D_h2)
            
            D_dosage_outcomes[treatment] = D_logits_treatment.squeeze(-1)
            
        # Combine outcomes from all treatments
        D_dosage_logits = torch.cat([D_dosage_outcomes[t] for t in range(self.num_treatments)], dim=-1)
        D_dosage_logits = D_dosage_logits.reshape(-1, self.num_treatments, self.num_dosage_samples)
        
        return D_dosage_logits, D_dosage_outcomes

    def treatment_discriminator(self, x, y, treatment_dosage_samples, treatment_dosage_mask, G_treatment_dosage_outcomes):
        # Move inputs to device
        x = x.to(self.device)
        y = y.to(self.device)
        treatment_dosage_samples = treatment_dosage_samples.to(self.device)
        treatment_dosage_mask = treatment_dosage_mask.to(self.device)
        
        # Get patient features representation
        patient_features_representation = self.D_treatment_features(x)
        
        D_treatment_outcomes = {}
        for treatment in range(self.num_treatments):
            treatment_mask = treatment_dosage_mask[:, treatment]
            treatment_dosages = treatment_dosage_samples[:, treatment]
            
            # Combine factual and generated outcomes
            factual_outcomes = treatment_mask * y
            counterfactual_outcomes = (1 - treatment_mask) * G_treatment_dosage_outcomes[treatment]
            treatment_outcomes = factual_outcomes + counterfactual_outcomes
            
            # Prepare inputs for invariant layer
            dosage_samples = treatment_dosages.unsqueeze(-1)  # [batch, num_dosage_samples, 1]
            dosage_potential_outcomes = treatment_outcomes.unsqueeze(-1)  # [batch, num_dosage_samples, 1]
            
            # Concatenate along last dimension
            inputs = torch.cat([dosage_samples, dosage_potential_outcomes], dim=-1)  # [batch, num_dosage_samples, 2]
            
            # Instanciation des couches équivariantes
            equivariant_layer_1 = equivariant_layer(inputs_dim=inputs.shape[-1], h_dim=self.h_inv_eqv_dim, layer_id=1, treatment_id=treatment)
            equivariant_layer_2 = equivariant_layer(inputs_dim=self.h_inv_eqv_dim, h_dim=self.h_inv_eqv_dim, layer_id=2, treatment_id=treatment)

            # Application des couches équivariantes
            D_h1 = torch.nn.functional.elu(
                equivariant_layer_1(inputs) + patient_features_representation.unsqueeze(1)  # Appel de l'instance avec les entrées
            )
            D_h2 = torch.nn.functional.elu(
                equivariant_layer_2(D_h1)  # Appel de l'instance avec D_h1
            )
            
            # Create a new invariant layer instance
            # Note: We create the instance with the correct dimension (not passing D_h2 itself)
            inv_layer = invariant_layer(input_dim=self.h_inv_eqv_dim, h_dim=self.h_inv_eqv_dim, treatment_id=treatment)
            
            # Then apply the layer to the data
            D_treatment_rep = inv_layer(D_h2)
            
            D_treatment_outcomes[treatment] = D_treatment_rep
            
        # Concatenate treatment representations
        D_treatment_representations = torch.cat([D_treatment_outcomes[t] for t in range(self.num_treatments)], dim=-1)
        
        # Concatenate with patient features
        D_shared_representation = torch.cat([D_treatment_representations, patient_features_representation], dim=-1)
        
        # Forward through shared representation layer
        D_treatment_rep_hidden = self.D_treatment_rep(D_shared_representation)
        
        # Final output layer
        D_treatment_logits = self.D_treatment_output(D_treatment_rep_hidden)
        
        return D_treatment_logits

    def inference(self, x, treatment_dosage_samples):
        # Move inputs to device
        x = x.to(self.device)
        treatment_dosage_samples = treatment_dosage_samples.to(self.device)
        
        # Forward through shared layer
        I_shared_out = self.I_shared(x)
        
        I_treatment_dosage_outcomes = {}

        for treatment in range(self.num_treatments):
            dosage_counterfactuals = []
            treatment_dosages = treatment_dosage_samples[:, treatment]
            
            for index in range(self.num_dosage_samples):
                dosage_sample = treatment_dosages[:, index].unsqueeze(-1)
                
                # Concatenate with shared representation
                input_counterfactual_dosage = torch.cat([I_shared_out, dosage_sample], dim=1)
                
                # Forward through treatment-specific layers
                treatment_layer_1_out = self.I_treatment_layers[treatment][0](input_counterfactual_dosage)
                treatment_layer_2_out = self.I_treatment_layers[treatment][1](treatment_layer_1_out)
                # In inference() method
                treatment_dosage_output = self.I_treatment_layers[treatment][2](treatment_layer_2_out)
                dosage_counterfactuals.append(treatment_dosage_output)
                
            # Stack outputs for all dosage samples
            I_treatment_dosage_outcomes[treatment] = torch.cat(dosage_counterfactuals, dim=1)
            
        # Combine outcomes from all treatments
        I_logits = torch.cat([I_treatment_dosage_outcomes[t] for t in range(self.num_treatments)], dim=1)
        I_logits = I_logits.reshape(-1, self.num_treatments, self.num_dosage_samples)
        
        return I_logits, I_treatment_dosage_outcomes
    

    def train(self, Train_X, Train_T, Train_D, Train_Y, verbose=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training SCIGAN generator and discriminator with device {device}")
        
        for it in tqdm(range(5000)):
            # PHASE 1: Train Generator
            # Sample mini-batch
            idx_mb = sample_X(Train_X, self.batch_size)
            X_mb = torch.tensor(Train_X[idx_mb, :], dtype=torch.float32).to(device)
            T_mb = torch.tensor(np.reshape(Train_T[idx_mb], [self.batch_size]), dtype=torch.float32).to(device)
            D_mb = torch.tensor(np.reshape(Train_D[idx_mb], [self.batch_size]), dtype=torch.float32).to(device)
            Y_mb = torch.tensor(np.reshape(Train_Y[idx_mb], [self.batch_size, 1]), dtype=torch.float32).to(device)
            Z_G_mb = sample_Z(self.batch_size, self.size_z).to(device)

            # Sample treatment dosages
            treatment_dosage_samples_gen = sample_dosages(self.batch_size, self.num_treatments, self.num_dosage_samples).to(device)
            treatment_dosage_samples_gen = treatment_dosage_samples_gen.clone()  # Éviter les modifications inplace
            factual_dosage_position = np.random.randint(self.num_dosage_samples, size=[self.batch_size])
            treatment_dosage_samples_gen[range(self.batch_size), T_mb.long(), factual_dosage_position] = D_mb.detach()
            
            # Create treatment dosage mask
            treatment_dosage_mask = torch.zeros(self.batch_size, self.num_treatments, self.num_dosage_samples, dtype=torch.float32).to(device)
            treatment_dosage_mask[range(self.batch_size), T_mb.long(), factual_dosage_position] = 1
            treatment_one_hot = torch.sum(treatment_dosage_mask, dim=-1)

            # Generator forward pass
            G_logits, G_treatment_dosage_outcomes = self.generator(X_mb, Y_mb, T_mb, D_mb, Z_G_mb, treatment_dosage_samples_gen)
            
            # Create detached copies for discriminator
            with torch.no_grad():
                G_treatment_detached = {}
                for k, v in G_treatment_dosage_outcomes.items():
                    G_treatment_detached[k] = v.detach()

            # Discriminator dosage forward pass with detached generator outputs
            D_dosage_logits, D_dosage_outcomes = self.dosage_discriminator(
                X_mb, Y_mb, treatment_dosage_samples_gen, treatment_dosage_mask, G_treatment_detached)
                
            # Treatment discriminator forward pass    
            D_treatment_logits = self.treatment_discriminator(
                X_mb, Y_mb, treatment_dosage_samples_gen, treatment_dosage_mask, G_treatment_detached)
            
            # Calculer les pertes comme dans TensorFlow:
            
            # 1. Dosage discriminator loss (comme dans TF)
            # Équivalent de factual_treatment_idx = tf.argmax(self.T, axis=1)
            factual_treatment_idx = T_mb.long()  # Déjà au bon format
            
            # Équivalent à tf.gather_nd(D_dosage_logits, idx)
            D_dosage_logits_factual_treatment = torch.zeros(self.batch_size, self.num_dosage_samples).to(device)
            for i in range(self.batch_size):
                treatment = int(factual_treatment_idx[i].item())
                D_dosage_logits_factual_treatment[i] = D_dosage_logits[i, treatment]
            
            # Dosage_Mask = tf.gather_nd(self.Treatment_Dosage_Mask, idx)
            Dosage_Mask = torch.zeros(self.batch_size, self.num_dosage_samples).to(device)
            for i in range(self.batch_size):
                treatment = int(factual_treatment_idx[i].item())
                Dosage_Mask[i] = treatment_dosage_mask[i, treatment]
            
            # BCE avec logits pour dosage discriminator
            D_dosage_loss = F.binary_cross_entropy_with_logits(
                input=D_dosage_logits_factual_treatment,
                target=Dosage_Mask
            )
            
            # 2. Treatment discriminator loss
            # Équivalent à tf.reduce_max(self.Treatment_Dosage_Mask, axis=-1)
            D_treatment_loss = F.binary_cross_entropy_with_logits(
                input=D_treatment_logits,
                target=treatment_one_hot
            )
            
            # 3. Combinaison des probabilités des discriminateurs pour la perte GAN
            # Appliquer sigmoid aux deux discriminateurs
            D_dosage_prob = torch.sigmoid(D_dosage_logits)
            
            # Équivalent à tf.tile(tf.expand_dims(D_treatment_logits, axis=-1), multiples=[1, 1, self.num_dosage_samples])
            D_treatment_prob_expanded = torch.sigmoid(D_treatment_logits).unsqueeze(-1).expand(-1, -1, self.num_dosage_samples)
            
            # Probabilités combinées comme dans le code TF
            D_combined_prob = D_dosage_prob * D_treatment_prob_expanded
            
            # Stabilité numérique
            D_combined_prob = torch.clamp(D_combined_prob, 1e-7, 1.0 - 1e-7)
            
            # Cross-entropy combinée pour discriminateurs
            D_combined_loss = torch.mean(
                treatment_dosage_mask * -torch.log(D_combined_prob) + 
                (1.0 - treatment_dosage_mask) * -torch.log(1.0 - D_combined_prob)
            )
            
            # 4. Generator loss (comme dans TF)
            G_loss_GAN = -D_combined_loss  # Même formulation que TF
            G_logit_factual = torch.sum(treatment_dosage_mask * G_logits, dim=[1, 2]).unsqueeze(-1)
            G_loss_R = torch.mean((Y_mb - G_logit_factual) ** 2)
            
            # Même formulation finale de la perte du générateur
            G_loss = self.alpha * torch.sqrt(G_loss_R) + G_loss_GAN
            
            # Update generator
            self.G_optimizer.zero_grad()
            G_loss.backward(retain_graph=True)  # retain_graph car on a besoin de ces calculs plus tard
            self.G_optimizer.step()

            # PHASE 2: Train Discriminators
            # Sample new mini-batch for discriminator
            idx_mb = sample_X(Train_X, self.batch_size)
            X_mb = torch.tensor(Train_X[idx_mb, :], dtype=torch.float32).to(device)
            T_mb = torch.tensor(np.reshape(Train_T[idx_mb], [self.batch_size]), dtype=torch.float32).to(device)
            D_mb = torch.tensor(np.reshape(Train_D[idx_mb], [self.batch_size]), dtype=torch.float32).to(device)
            Y_mb = torch.tensor(np.reshape(Train_Y[idx_mb], [self.batch_size, 1]), dtype=torch.float32).to(device)
            Z_G_mb = sample_Z(self.batch_size, self.size_z).to(device)
            
            # Sample treatment dosages
            treatment_dosage_samples_disc = sample_dosages(self.batch_size, self.num_treatments, self.num_dosage_samples).to(device)
            treatment_dosage_samples_disc = treatment_dosage_samples_disc.clone()  # Pour éviter inplace operation
            factual_dosage_position = np.random.randint(self.num_dosage_samples, size=[self.batch_size])
            treatment_dosage_samples_disc[range(self.batch_size), T_mb.long(), factual_dosage_position] = D_mb.detach()
            
            # Create treatment dosage mask
            treatment_dosage_mask = torch.zeros(self.batch_size, self.num_treatments, self.num_dosage_samples, dtype=torch.float32).to(device)
            treatment_dosage_mask[range(self.batch_size), T_mb.long(), factual_dosage_position] = 1
            treatment_one_hot = torch.sum(treatment_dosage_mask, dim=-1)
            
            # Generate outcomes with no_grad to avoid backprop through generator
            with torch.no_grad():
                G_logits, G_treatment_dosage_outcomes = self.generator(X_mb, Y_mb, T_mb, D_mb, Z_G_mb, treatment_dosage_samples_disc)

            # Train dosage discriminator - même calcul de perte que pour le générateur
            D_dosage_logits, D_dosage_outcomes = self.dosage_discriminator(
                X_mb, Y_mb, treatment_dosage_samples_disc, treatment_dosage_mask, G_treatment_dosage_outcomes)
            
            # Calculer la perte du dosage discriminator avec la formulation TensorFlow
            factual_treatment_idx = T_mb.long()
            D_dosage_logits_factual_treatment = torch.zeros(self.batch_size, self.num_dosage_samples).to(device)
            Dosage_Mask = torch.zeros(self.batch_size, self.num_dosage_samples).to(device)
            
            for i in range(self.batch_size):
                treatment = int(factual_treatment_idx[i].item())
                D_dosage_logits_factual_treatment[i] = D_dosage_logits[i, treatment]
                Dosage_Mask[i] = treatment_dosage_mask[i, treatment]
                
            D_dosage_loss = F.binary_cross_entropy_with_logits(
                input=D_dosage_logits_factual_treatment,
                target=Dosage_Mask
            )
            
            self.D_dosage_optimizer.zero_grad()
            D_dosage_loss.backward(retain_graph=True)
            self.D_dosage_optimizer.step()

            # Train treatment discriminator
            D_treatment_logits = self.treatment_discriminator(
                X_mb, Y_mb, treatment_dosage_samples_disc, treatment_dosage_mask, G_treatment_dosage_outcomes)
            
            D_treatment_loss = F.binary_cross_entropy_with_logits(
                input=D_treatment_logits,
                target=treatment_one_hot
            )
            
            self.D_treatment_optimizer.zero_grad()
            D_treatment_loss.backward()
            self.D_treatment_optimizer.step()

            # Debug output
            if it % 500 == 0 and verbose:
                print('Iter: {}'.format(it))
                print('D_loss_treatments: {:.4}'.format(D_treatment_loss.item()))
                print('D_loss_dosages: {:.4}'.format(D_dosage_loss.item()))
                print('G_loss: {:.4}'.format(G_loss.item()))
                print()

        # PHASE 3: Train Inference Network
        print("Training inference network.")
        
        for it in tqdm(range(10000)):
            # Sample mini-batch
            idx_mb = sample_X(Train_X, self.batch_size)
            X_mb = torch.tensor(Train_X[idx_mb, :], dtype=torch.float32).to(device)
            T_mb = torch.tensor(np.reshape(Train_T[idx_mb], [self.batch_size]), dtype=torch.float32).to(device)
            D_mb = torch.tensor(np.reshape(Train_D[idx_mb], [self.batch_size]), dtype=torch.float32).to(device)
            Y_mb = torch.tensor(np.reshape(Train_Y[idx_mb], [self.batch_size, 1]), dtype=torch.float32).to(device)
            
            # Sample treatment dosages
            treatment_dosage_samples = sample_dosages(self.batch_size, self.num_treatments, self.num_dosage_samples).to(device)
            treatment_dosage_samples = treatment_dosage_samples.clone()  # Éviter inplace operation
            factual_dosage_position = np.random.randint(self.num_dosage_samples, size=[self.batch_size])
            treatment_dosage_samples[range(self.batch_size), T_mb.long(), factual_dosage_position] = D_mb.detach()
            
            # Create treatment dosage mask
            treatment_dosage_mask = torch.zeros(self.batch_size, self.num_treatments, self.num_dosage_samples, dtype=torch.float32).to(device)
            treatment_dosage_mask[range(self.batch_size), T_mb.long(), factual_dosage_position] = 1
            
            # Generate outcomes with generator (pour I_loss1)
            with torch.no_grad():
                G_logits, _ = self.generator(X_mb, Y_mb, T_mb, D_mb, 
                                            sample_Z(self.batch_size, self.size_z).to(device), 
                                            treatment_dosage_samples)
                
            # Forward pass of inference network
            I_logits, _ = self.inference(X_mb, treatment_dosage_samples)
            
            # I_logit_factual comme dans TensorFlow
            I_logit_factual = torch.sum(treatment_dosage_mask * I_logits, dim=[1, 2]).unsqueeze(-1)
            
            # Calculer les pertes d'inférence comme dans TensorFlow
            I_loss1 = torch.mean((G_logits.detach() - I_logits)**2)  # MSE entre générateur et inférence
            I_loss2 = torch.mean((Y_mb - I_logit_factual)**2)  # MSE entre sortie factuelle et cible
            
            # Même formulation que TensorFlow
            I_loss = torch.sqrt(I_loss1) + torch.sqrt(I_loss2)
            
            # Update inference network
            self.I_optimizer.zero_grad()
            I_loss.backward()
            self.I_optimizer.step()
            
            # Debug output
            if it % 1000 == 0 and verbose:
                print('Iter: {}'.format(it))
                print('I_loss: {:.4}'.format(I_loss.item()))
                print('I_loss1 (Generator MSE): {:.4}'.format(I_loss1.item()))
                print('I_loss2 (Factual MSE): {:.4}'.format(I_loss2.item()))
                print()

        # Save the model
        inference_state = {
            'shared': self.I_shared.state_dict(),
            'treatment_layers': [layer.state_dict() for layer in self.I_treatment_layers]
        }
        torch.save(inference_state, self.export_dir + '/inference_network.pth')
