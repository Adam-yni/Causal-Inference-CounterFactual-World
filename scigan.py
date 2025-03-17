import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# Assuming these functions are already defined in utils.model_utils
from utils import equivariant_layer, invariant_layer, sample_dosages, sample_X, sample_Z

# Désactiver la détection d'anomalie en production pour la performance
torch.autograd.set_detect_anomaly(False)

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
        
        # Déplacer tous les modèles sur le GPU en une seule fois
        self._move_models_to_device()
        
        # Précréer les couches équivariantes et invariantes
        self._build_equivariant_invariant_layers()
        
        # Initialize optimizers
        self.G_optimizer = optim.Adam(self.generator_params, lr=0.001)
        self.D_dosage_optimizer = optim.Adam(self.dosage_discriminator_params, lr=0.001)
        self.D_treatment_optimizer = optim.Adam(self.treatment_discriminator_params, lr=0.001)
        self.I_optimizer = optim.Adam(self.inference_params, lr=0.001)

    def _move_models_to_device(self):
        """Déplace tous les modèles sur le device en une seule fois"""
        self.G_shared = self.G_shared.to(self.device)
        self.G_treatment_layers = self.G_treatment_layers.to(self.device)
        self.D_dosage_features = self.D_dosage_features.to(self.device)
        self.D_dosage_outputs = self.D_dosage_outputs.to(self.device)
        self.D_treatment_features = self.D_treatment_features.to(self.device)
        self.D_treatment_rep = self.D_treatment_rep.to(self.device)
        self.D_treatment_output = self.D_treatment_output.to(self.device)
        self.I_shared = self.I_shared.to(self.device)
        self.I_treatment_layers = self.I_treatment_layers.to(self.device)

    def _build_equivariant_invariant_layers(self):
        """Précalcule et stocke les couches équivariantes et invariantes"""
        # Équivariant layers pour dosage discriminator
        self.eqv_layers_dosage = {}
        # Équivariant layers pour treatment discriminator
        self.eqv_layers_treatment = {}
        # Invariant layers
        self.inv_layers = {}
        
        for treatment in range(self.num_treatments):
            # Pour dosage discriminator
            self.eqv_layers_dosage[treatment] = [
                equivariant_layer(inputs_dim=2, h_dim=self.h_inv_eqv_dim, layer_id=1, treatment_id=treatment).to(self.device),
                equivariant_layer(inputs_dim=self.h_inv_eqv_dim, h_dim=self.h_inv_eqv_dim, layer_id=2, treatment_id=treatment).to(self.device)
            ]
            
            # Pour treatment discriminator
            self.eqv_layers_treatment[treatment] = [
                equivariant_layer(inputs_dim=2, h_dim=self.h_inv_eqv_dim, layer_id=1, treatment_id=treatment).to(self.device),
                equivariant_layer(inputs_dim=self.h_inv_eqv_dim, h_dim=self.h_inv_eqv_dim, layer_id=2, treatment_id=treatment).to(self.device)
            ]
            
            # Invariant layer
            self.inv_layers[treatment] = invariant_layer(
                input_dim=self.h_inv_eqv_dim, h_dim=self.h_inv_eqv_dim, treatment_id=treatment
            ).to(self.device)

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
        
        # Get patient features representation - pas besoin de .to(self.device)
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
            
            # Utiliser les couches précalculées
            D_h1 = torch.nn.functional.elu(
                self.eqv_layers_dosage[treatment][0](inputs) + patient_features_representation
            )
            D_h2 = torch.nn.functional.elu(
                self.eqv_layers_dosage[treatment][1](D_h1)
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
            
            # Prepare inputs for equivariant layer
            dosage_samples = treatment_dosages.unsqueeze(-1)  # [batch, num_dosage_samples, 1]
            dosage_potential_outcomes = treatment_outcomes.unsqueeze(-1)  # [batch, num_dosage_samples, 1]
            
            # Concatenate along last dimension
            inputs = torch.cat([dosage_samples, dosage_potential_outcomes], dim=-1)  # [batch, num_dosage_samples, 2]
            
            # Utiliser les couches précalculées
            D_h1 = torch.nn.functional.elu(
                self.eqv_layers_treatment[treatment][0](inputs) + patient_features_representation.unsqueeze(1)
            )
            D_h2 = torch.nn.functional.elu(
                self.eqv_layers_treatment[treatment][1](D_h1)
            )
            
            # Utiliser la couche invariante précalculée
            D_treatment_rep = self.inv_layers[treatment](D_h2)
            
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
                treatment_dosage_output = self.I_treatment_layers[treatment][2](treatment_layer_2_out)
                dosage_counterfactuals.append(treatment_dosage_output)
                
            # Stack outputs for all dosage samples
            I_treatment_dosage_outcomes[treatment] = torch.cat(dosage_counterfactuals, dim=1)
            
        # Combine outcomes from all treatments
        I_logits = torch.cat([I_treatment_dosage_outcomes[t] for t in range(self.num_treatments)], dim=1)
        I_logits = I_logits.reshape(-1, self.num_treatments, self.num_dosage_samples)
        
        return I_logits, I_treatment_dosage_outcomes
    

    def train(self, Train_X, Train_T, Train_D, Train_Y, verbose=False):
        print(f"Training SCIGAN generator and discriminator with device {self.device}")
        
        # Convertir les données en tenseurs PyTorch une seule fois
        Train_X_tensor = torch.tensor(Train_X, dtype=torch.float32)
        Train_T_tensor = torch.tensor(np.reshape(Train_T, [-1]), dtype=torch.float32)
        Train_D_tensor = torch.tensor(np.reshape(Train_D, [-1]), dtype=torch.float32)
        Train_Y_tensor = torch.tensor(np.reshape(Train_Y, [-1, 1]), dtype=torch.float32)
        
        for it in tqdm(range(5000)):
            # PHASE 1: Train Generator
            # Sample mini-batch
            idx_mb = sample_X(Train_X, self.batch_size)
            X_mb = Train_X_tensor[idx_mb].to(self.device)
            T_mb = Train_T_tensor[idx_mb].to(self.device)
            D_mb = Train_D_tensor[idx_mb].to(self.device)
            Y_mb = Train_Y_tensor[idx_mb].to(self.device)
            Z_G_mb = sample_Z(self.batch_size, self.size_z).to(self.device)

            # Sample treatment dosages - optimisé pour l'utilisation du GPU
            treatment_dosage_samples_gen = sample_dosages(self.batch_size, self.num_treatments, self.num_dosage_samples).to(self.device)
            
            # Convertir factual_dosage_position en tenseur GPU
            factual_dosage_position = torch.tensor(
                np.random.randint(self.num_dosage_samples, size=[self.batch_size]),
                dtype=torch.long
            ).to(self.device)
            
            # Indexation vectorisée via des tenseurs batch
            batch_indices = torch.arange(self.batch_size, device=self.device)
            treatment_indices = T_mb.long()
            
            # Utiliser une approche vectorisée pour l'indexation
            treatment_dosage_samples_gen[batch_indices, treatment_indices, factual_dosage_position] = D_mb
            
            # Create treatment dosage mask - également vectorisé
            treatment_dosage_mask = torch.zeros(
                self.batch_size, self.num_treatments, self.num_dosage_samples, 
                dtype=torch.float32, device=self.device
            )
            treatment_dosage_mask[batch_indices, treatment_indices, factual_dosage_position] = 1
            treatment_one_hot = torch.sum(treatment_dosage_mask, dim=-1)

            # Generator forward pass
            G_logits, G_treatment_dosage_outcomes = self.generator(X_mb, Y_mb, T_mb, D_mb, Z_G_mb, treatment_dosage_samples_gen)
            
            # Create detached copies for discriminator
            with torch.no_grad():
                G_treatment_detached = {k: v.detach() for k, v in G_treatment_dosage_outcomes.items()}

            # Discriminator dosage forward pass with detached generator outputs
            D_dosage_logits, D_dosage_outcomes = self.dosage_discriminator(
                X_mb, Y_mb, treatment_dosage_samples_gen, treatment_dosage_mask, G_treatment_detached)
                
            # Treatment discriminator forward pass    
            D_treatment_logits = self.treatment_discriminator(
                X_mb, Y_mb, treatment_dosage_samples_gen, treatment_dosage_mask, G_treatment_detached)
            
            # Extraction des logits via gather pour éviter les boucles
            D_dosage_logits_factual_treatment = D_dosage_logits.gather(
                1, 
                treatment_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.num_dosage_samples)
            ).squeeze(1)
            
            Dosage_Mask = treatment_dosage_mask.gather(
                1,
                treatment_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.num_dosage_samples)
            ).squeeze(1)
            
            # BCE avec logits pour dosage discriminator
            D_dosage_loss = F.binary_cross_entropy_with_logits(
                input=D_dosage_logits_factual_treatment,
                target=Dosage_Mask
            )
            
            # Treatment discriminator loss
            D_treatment_loss = F.binary_cross_entropy_with_logits(
                input=D_treatment_logits,
                target=treatment_one_hot
            )
            
            # Combinaison des probabilités des discriminateurs pour la perte GAN
            D_dosage_prob = torch.sigmoid(D_dosage_logits)
            D_treatment_prob_expanded = torch.sigmoid(D_treatment_logits).unsqueeze(-1).expand(-1, -1, self.num_dosage_samples)
            D_combined_prob = D_dosage_prob * D_treatment_prob_expanded
            
            # Cross-entropy combinée pour discriminateurs
            D_combined_loss = torch.mean(
                treatment_dosage_mask * -torch.log(D_combined_prob + 1e-7) + 
                (1.0 - treatment_dosage_mask) * -torch.log(1.0 - D_combined_prob + 1e-7)
            )
            
            # Generator loss
            G_loss_GAN = -D_combined_loss
            G_logit_factual = torch.sum(treatment_dosage_mask * G_logits, dim=[1, 2]).unsqueeze(-1)
            G_loss_R = torch.mean((Y_mb - G_logit_factual) ** 2)
            
            G_loss = self.alpha * torch.sqrt(G_loss_R) + G_loss_GAN
            
            # Update generator
            self.G_optimizer.zero_grad()
            G_loss.backward(retain_graph=True)
            self.G_optimizer.step()

            # PHASE 2: Train Discriminators
            # Sample new mini-batch for discriminator
            idx_mb = sample_X(Train_X, self.batch_size)
            X_mb = Train_X_tensor[idx_mb].to(self.device)
            T_mb = Train_T_tensor[idx_mb].to(self.device)
            D_mb = Train_D_tensor[idx_mb].to(self.device)
            Y_mb = Train_Y_tensor[idx_mb].to(self.device)
            Z_G_mb = sample_Z(self.batch_size, self.size_z).to(self.device)
            
            # Sample treatment dosages - même approche vectorisée
            treatment_dosage_samples_disc = sample_dosages(self.batch_size, self.num_treatments, self.num_dosage_samples).to(self.device)
            
            factual_dosage_position = torch.tensor(
                np.random.randint(self.num_dosage_samples, size=[self.batch_size]),
                dtype=torch.long
            ).to(self.device)
            
            batch_indices = torch.arange(self.batch_size, device=self.device)
            treatment_indices = T_mb.long()
            
            treatment_dosage_samples_disc[batch_indices, treatment_indices, factual_dosage_position] = D_mb
            
            treatment_dosage_mask = torch.zeros(
                self.batch_size, self.num_treatments, self.num_dosage_samples, 
                dtype=torch.float32, device=self.device
            )
            treatment_dosage_mask[batch_indices, treatment_indices, factual_dosage_position] = 1
            treatment_one_hot = torch.sum(treatment_dosage_mask, dim=-1)
            
            # Generate outcomes with no_grad to avoid backprop through generator
            with torch.no_grad():
                G_logits, G_treatment_dosage_outcomes = self.generator(X_mb, Y_mb, T_mb, D_mb, Z_G_mb, treatment_dosage_samples_disc)

            # Train dosage discriminator
            D_dosage_logits, D_dosage_outcomes = self.dosage_discriminator(
                X_mb, Y_mb, treatment_dosage_samples_disc, treatment_dosage_mask, G_treatment_dosage_outcomes)
            
            # Utiliser gather pour extraction vectorisée
            D_dosage_logits_factual_treatment = D_dosage_logits.gather(
                1, 
                treatment_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.num_dosage_samples)
            ).squeeze(1)
            
            Dosage_Mask = treatment_dosage_mask.gather(
                1,
                treatment_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.num_dosage_samples)
            ).squeeze(1)
                
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
            X_mb = Train_X_tensor[idx_mb].to(self.device)
            T_mb = Train_T_tensor[idx_mb].to(self.device)
            D_mb = Train_D_tensor[idx_mb].to(self.device)
            Y_mb = Train_Y_tensor[idx_mb].to(self.device)
            
            # Sample treatment dosages - même approche vectorisée
            treatment_dosage_samples = sample_dosages(self.batch_size, self.num_treatments, self.num_dosage_samples).to(self.device)
            
            factual_dosage_position = torch.tensor(
                np.random.randint(self.num_dosage_samples, size=[self.batch_size]),
                dtype=torch.long
            ).to(self.device)
            
            batch_indices = torch.arange(self.batch_size, device=self.device)
            treatment_indices = T_mb.long()
            
            treatment_dosage_samples[batch_indices, treatment_indices, factual_dosage_position] = D_mb
            
            treatment_dosage_mask = torch.zeros(
                self.batch_size, self.num_treatments, self.num_dosage_samples, 
                dtype=torch.float32, device=self.device
            )
            treatment_dosage_mask[batch_indices, treatment_indices, factual_dosage_position] = 1
            
            # Generate outcomes with generator
            with torch.no_grad():
                G_logits, _ = self.generator(X_mb, Y_mb, T_mb, D_mb, 
                                            sample_Z(self.batch_size, self.size_z).to(self.device), 
                                            treatment_dosage_samples)
                
            # Forward pass of inference network
            I_logits, _ = self.inference(X_mb, treatment_dosage_samples)
            
            # I_logit_factual comme dans TensorFlow
            I_logit_factual = torch.sum(treatment_dosage_mask * I_logits, dim=[1, 2]).unsqueeze(-1)
            
            # Calculer les pertes d'inférence comme dans TensorFlow# filepath: c:\Users\adamy\Desktop\RTE\code\exemples\test1\scigan.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# Assuming these functions are already defined in utils.model_utils
from utils import equivariant_layer, invariant_layer, sample_dosages, sample_X, sample_Z

# Désactiver la détection d'anomalie en production pour la performance
torch.autograd.set_detect_anomaly(False)

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
        
        # Déplacer tous les modèles sur le GPU en une seule fois
        self._move_models_to_device()
        
        # Précréer les couches équivariantes et invariantes
        self._build_equivariant_invariant_layers()
        
        # Initialize optimizers
        self.G_optimizer = optim.Adam(self.generator_params, lr=0.001)
        self.D_dosage_optimizer = optim.Adam(self.dosage_discriminator_params, lr=0.001)
        self.D_treatment_optimizer = optim.Adam(self.treatment_discriminator_params, lr=0.001)
        self.I_optimizer = optim.Adam(self.inference_params, lr=0.001)

    def _move_models_to_device(self):
        """Déplace tous les modèles sur le device en une seule fois"""
        self.G_shared = self.G_shared.to(self.device)
        self.G_treatment_layers = self.G_treatment_layers.to(self.device)
        self.D_dosage_features = self.D_dosage_features.to(self.device)
        self.D_dosage_outputs = self.D_dosage_outputs.to(self.device)
        self.D_treatment_features = self.D_treatment_features.to(self.device)
        self.D_treatment_rep = self.D_treatment_rep.to(self.device)
        self.D_treatment_output = self.D_treatment_output.to(self.device)
        self.I_shared = self.I_shared.to(self.device)
        self.I_treatment_layers = self.I_treatment_layers.to(self.device)

    def _build_equivariant_invariant_layers(self):
        """Précalcule et stocke les couches équivariantes et invariantes"""
        # Équivariant layers pour dosage discriminator
        self.eqv_layers_dosage = {}
        # Équivariant layers pour treatment discriminator
        self.eqv_layers_treatment = {}
        # Invariant layers
        self.inv_layers = {}
        
        for treatment in range(self.num_treatments):
            # Pour dosage discriminator
            self.eqv_layers_dosage[treatment] = [
                equivariant_layer(inputs_dim=2, h_dim=self.h_inv_eqv_dim, layer_id=1, treatment_id=treatment).to(self.device),
                equivariant_layer(inputs_dim=self.h_inv_eqv_dim, h_dim=self.h_inv_eqv_dim, layer_id=2, treatment_id=treatment).to(self.device)
            ]
            
            # Pour treatment discriminator
            self.eqv_layers_treatment[treatment] = [
                equivariant_layer(inputs_dim=2, h_dim=self.h_inv_eqv_dim, layer_id=1, treatment_id=treatment).to(self.device),
                equivariant_layer(inputs_dim=self.h_inv_eqv_dim, h_dim=self.h_inv_eqv_dim, layer_id=2, treatment_id=treatment).to(self.device)
            ]
            
            # Invariant layer
            self.inv_layers[treatment] = invariant_layer(
                input_dim=self.h_inv_eqv_dim, h_dim=self.h_inv_eqv_dim, treatment_id=treatment
            ).to(self.device)

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
        
        # Get patient features representation - pas besoin de .to(self.device)
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
            
            # Utiliser les couches précalculées
            D_h1 = torch.nn.functional.elu(
                self.eqv_layers_dosage[treatment][0](inputs) + patient_features_representation
            )
            D_h2 = torch.nn.functional.elu(
                self.eqv_layers_dosage[treatment][1](D_h1)
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
            
            # Prepare inputs for equivariant layer
            dosage_samples = treatment_dosages.unsqueeze(-1)  # [batch, num_dosage_samples, 1]
            dosage_potential_outcomes = treatment_outcomes.unsqueeze(-1)  # [batch, num_dosage_samples, 1]
            
            # Concatenate along last dimension
            inputs = torch.cat([dosage_samples, dosage_potential_outcomes], dim=-1)  # [batch, num_dosage_samples, 2]
            
            # Utiliser les couches précalculées
            D_h1 = torch.nn.functional.elu(
                self.eqv_layers_treatment[treatment][0](inputs) + patient_features_representation.unsqueeze(1)
            )
            D_h2 = torch.nn.functional.elu(
                self.eqv_layers_treatment[treatment][1](D_h1)
            )
            
            # Utiliser la couche invariante précalculée
            D_treatment_rep = self.inv_layers[treatment](D_h2)
            
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
                treatment_dosage_output = self.I_treatment_layers[treatment][2](treatment_layer_2_out)
                dosage_counterfactuals.append(treatment_dosage_output)
                
            # Stack outputs for all dosage samples
            I_treatment_dosage_outcomes[treatment] = torch.cat(dosage_counterfactuals, dim=1)
            
        # Combine outcomes from all treatments
        I_logits = torch.cat([I_treatment_dosage_outcomes[t] for t in range(self.num_treatments)], dim=1)
        I_logits = I_logits.reshape(-1, self.num_treatments, self.num_dosage_samples)
        
        return I_logits, I_treatment_dosage_outcomes
    

    def train(self, Train_X, Train_T, Train_D, Train_Y, verbose=False):
        print(f"Training SCIGAN generator and discriminator with device {self.device}")
        
        # Convertir les données en tenseurs PyTorch une seule fois
        Train_X_tensor = torch.tensor(Train_X, dtype=torch.float32)
        Train_T_tensor = torch.tensor(np.reshape(Train_T, [-1]), dtype=torch.float32)
        Train_D_tensor = torch.tensor(np.reshape(Train_D, [-1]), dtype=torch.float32)
        Train_Y_tensor = torch.tensor(np.reshape(Train_Y, [-1, 1]), dtype=torch.float32)
        
        for it in tqdm(range(5000)):
            # PHASE 1: Train Generator
            # Sample mini-batch
            idx_mb = sample_X(Train_X, self.batch_size)
            X_mb = Train_X_tensor[idx_mb].to(self.device)
            T_mb = Train_T_tensor[idx_mb].to(self.device)
            D_mb = Train_D_tensor[idx_mb].to(self.device)
            Y_mb = Train_Y_tensor[idx_mb].to(self.device)
            Z_G_mb = sample_Z(self.batch_size, self.size_z).to(self.device)

            # Sample treatment dosages - optimisé pour l'utilisation du GPU
            treatment_dosage_samples_gen = sample_dosages(self.batch_size, self.num_treatments, self.num_dosage_samples).to(self.device)
            
            # Convertir factual_dosage_position en tenseur GPU
            factual_dosage_position = torch.tensor(
                np.random.randint(self.num_dosage_samples, size=[self.batch_size]),
                dtype=torch.long
            ).to(self.device)
            
            # Indexation vectorisée via des tenseurs batch
            batch_indices = torch.arange(self.batch_size, device=self.device)
            treatment_indices = T_mb.long()
            
            # Utiliser une approche vectorisée pour l'indexation
            treatment_dosage_samples_gen[batch_indices, treatment_indices, factual_dosage_position] = D_mb
            
            # Create treatment dosage mask - également vectorisé
            treatment_dosage_mask = torch.zeros(
                self.batch_size, self.num_treatments, self.num_dosage_samples, 
                dtype=torch.float32, device=self.device
            )
            treatment_dosage_mask[batch_indices, treatment_indices, factual_dosage_position] = 1
            treatment_one_hot = torch.sum(treatment_dosage_mask, dim=-1)

            # Generator forward pass
            G_logits, G_treatment_dosage_outcomes = self.generator(X_mb, Y_mb, T_mb, D_mb, Z_G_mb, treatment_dosage_samples_gen)
            
            # Create detached copies for discriminator
            with torch.no_grad():
                G_treatment_detached = {k: v.detach() for k, v in G_treatment_dosage_outcomes.items()}

            # Discriminator dosage forward pass with detached generator outputs
            D_dosage_logits, D_dosage_outcomes = self.dosage_discriminator(
                X_mb, Y_mb, treatment_dosage_samples_gen, treatment_dosage_mask, G_treatment_detached)
                
            # Treatment discriminator forward pass    
            D_treatment_logits = self.treatment_discriminator(
                X_mb, Y_mb, treatment_dosage_samples_gen, treatment_dosage_mask, G_treatment_detached)
            
            # Extraction des logits via gather pour éviter les boucles
            D_dosage_logits_factual_treatment = D_dosage_logits.gather(
                1, 
                treatment_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.num_dosage_samples)
            ).squeeze(1)
            
            Dosage_Mask = treatment_dosage_mask.gather(
                1,
                treatment_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.num_dosage_samples)
            ).squeeze(1)
            
            # BCE avec logits pour dosage discriminator
            D_dosage_loss = F.binary_cross_entropy_with_logits(
                input=D_dosage_logits_factual_treatment,
                target=Dosage_Mask
            )
            
            # Treatment discriminator loss
            D_treatment_loss = F.binary_cross_entropy_with_logits(
                input=D_treatment_logits,
                target=treatment_one_hot
            )
            
            # Combinaison des probabilités des discriminateurs pour la perte GAN
            D_dosage_prob = torch.sigmoid(D_dosage_logits)
            D_treatment_prob_expanded = torch.sigmoid(D_treatment_logits).unsqueeze(-1).expand(-1, -1, self.num_dosage_samples)
            D_combined_prob = D_dosage_prob * D_treatment_prob_expanded
            
            # Cross-entropy combinée pour discriminateurs
            D_combined_loss = torch.mean(
                treatment_dosage_mask * -torch.log(D_combined_prob + 1e-7) + 
                (1.0 - treatment_dosage_mask) * -torch.log(1.0 - D_combined_prob + 1e-7)
            )
            
            # Generator loss
            G_loss_GAN = -D_combined_loss
            G_logit_factual = torch.sum(treatment_dosage_mask * G_logits, dim=[1, 2]).unsqueeze(-1)
            G_loss_R = torch.mean((Y_mb - G_logit_factual) ** 2)
            
            G_loss = self.alpha * torch.sqrt(G_loss_R) + G_loss_GAN
            
            # Update generator
            self.G_optimizer.zero_grad()
            G_loss.backward(retain_graph=True)
            self.G_optimizer.step()

            # PHASE 2: Train Discriminators
            # Sample new mini-batch for discriminator
            idx_mb = sample_X(Train_X, self.batch_size)
            X_mb = Train_X_tensor[idx_mb].to(self.device)
            T_mb = Train_T_tensor[idx_mb].to(self.device)
            D_mb = Train_D_tensor[idx_mb].to(self.device)
            Y_mb = Train_Y_tensor[idx_mb].to(self.device)
            Z_G_mb = sample_Z(self.batch_size, self.size_z).to(self.device)
            
            # Sample treatment dosages - même approche vectorisée
            treatment_dosage_samples_disc = sample_dosages(self.batch_size, self.num_treatments, self.num_dosage_samples).to(self.device)
            
            factual_dosage_position = torch.tensor(
                np.random.randint(self.num_dosage_samples, size=[self.batch_size]),
                dtype=torch.long
            ).to(self.device)
            
            batch_indices = torch.arange(self.batch_size, device=self.device)
            treatment_indices = T_mb.long()
            
            treatment_dosage_samples_disc[batch_indices, treatment_indices, factual_dosage_position] = D_mb
            
            treatment_dosage_mask = torch.zeros(
                self.batch_size, self.num_treatments, self.num_dosage_samples, 
                dtype=torch.float32, device=self.device
            )
            treatment_dosage_mask[batch_indices, treatment_indices, factual_dosage_position] = 1
            treatment_one_hot = torch.sum(treatment_dosage_mask, dim=-1)
            
            # Generate outcomes with no_grad to avoid backprop through generator
            with torch.no_grad():
                G_logits, G_treatment_dosage_outcomes = self.generator(X_mb, Y_mb, T_mb, D_mb, Z_G_mb, treatment_dosage_samples_disc)

            # Train dosage discriminator
            D_dosage_logits, D_dosage_outcomes = self.dosage_discriminator(
                X_mb, Y_mb, treatment_dosage_samples_disc, treatment_dosage_mask, G_treatment_dosage_outcomes)
            
            # Utiliser gather pour extraction vectorisée
            D_dosage_logits_factual_treatment = D_dosage_logits.gather(
                1, 
                treatment_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.num_dosage_samples)
            ).squeeze(1)
            
            Dosage_Mask = treatment_dosage_mask.gather(
                1,
                treatment_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.num_dosage_samples)
            ).squeeze(1)
                
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
            X_mb = Train_X_tensor[idx_mb].to(self.device)
            T_mb = Train_T_tensor[idx_mb].to(self.device)
            D_mb = Train_D_tensor[idx_mb].to(self.device)
            Y_mb = Train_Y_tensor[idx_mb].to(self.device)
            
            # Sample treatment dosages - même approche vectorisée
            treatment_dosage_samples = sample_dosages(self.batch_size, self.num_treatments, self.num_dosage_samples).to(self.device)
            
            factual_dosage_position = torch.tensor(
                np.random.randint(self.num_dosage_samples, size=[self.batch_size]),
                dtype=torch.long
            ).to(self.device)
            
            batch_indices = torch.arange(self.batch_size, device=self.device)
            treatment_indices = T_mb.long()
            
            treatment_dosage_samples[batch_indices, treatment_indices, factual_dosage_position] = D_mb
            
            treatment_dosage_mask = torch.zeros(
                self.batch_size, self.num_treatments, self.num_dosage_samples, 
                dtype=torch.float32, device=self.device
            )
            treatment_dosage_mask[batch_indices, treatment_indices, factual_dosage_position] = 1
            
            # Generate outcomes with generator
            with torch.no_grad():
                G_logits, _ = self.generator(X_mb, Y_mb, T_mb, D_mb, 
                                            sample_Z(self.batch_size, self.size_z).to(self.device), 
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
