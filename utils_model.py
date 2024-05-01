import torch
import torch.nn as nn

# Teacher Model Components
class EncoderEv(nn.Module):
    def __init__(self, input_dim):
        super(EncoderEv, self).__init__()
        self.gen = nn.Sequential(
            nn.Conv2d(input_dim, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.gen(x)
        return x

class DecoderDv(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(DecoderDv, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 30, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(30)
        self.deconv4 = nn.ConvTranspose2d(30, output_dim, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(28)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        return x


class DiscriminatorC(nn.Module):
    def __init__(self, input_dim):
        super(DiscriminatorC, self).__init__()
        self.f0 = nn.Sequential(
            nn.Conv2d(input_dim, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU()
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.out(x)
        return x


# Student Model Components
class EncoderEs(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EncoderEs, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.conv = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = self.relu(h)
        h = h[-1]  # Get the hidden state of the last LSTM unit
        h = h.unsqueeze(2).unsqueeze(3)  # Add dimensions for 2D convolution
        v = self.conv(h)
        return v

class EnEv(nn.Module):
    def __init__(self, embedding_dim, input_dim):
        super(EnEv, self).__init__()
        self.L1 = nn.Sequential(
            nn.Linear(input_dim, 25, dtype=torch.double),
            nn.LeakyReLU(),
            nn.Linear(25, 64, dtype=torch.double),
            nn.LeakyReLU(),
            nn.Linear(64, embedding_dim, dtype=torch.double),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        y = self.L1(x)
        # y = self.L1(x.to(self.L1[0].weight.dtype))

        return y
class EnDv(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(EnDv, self).__init__()
        self.L2 = nn.Sequential(
            nn.Linear(embedding_dim, 64, dtype=torch.double),
            nn.LeakyReLU(),
            nn.Linear(64, 25, dtype=torch.double),
            nn.LeakyReLU(),
            nn.Linear(25, output_dim, dtype=torch.double),
            nn.Sigmoid()  # ？
        )

    def forward(self, x):
        x = self.L2(x)

        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 对应Squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # 对应Excitation操作
        return x * y.expand_as(x)

# pretrain TeacherModel
class TeacherModel_G(nn.Module):
    def __init__(self, ev_input_dim, ev_latent_dim, dv_output_dim):
        super(TeacherModel_G, self).__init__()
        self.teacher_encoder_ev = EncoderEv(ev_input_dim).double()
        self.teacher_decoder_dv = DecoderDv(ev_latent_dim, dv_output_dim).double()
        self.selayer = SELayer(ev_latent_dim).double()

    def forward(self, f):
        z = self.teacher_encoder_ev(f)
        z_atti = self.selayer(z)
        y = self.teacher_decoder_dv(z_atti)

        return y

class TeacherModel_D(nn.Module):
    def __init__(self, ev_input_dim):
        super(TeacherModel_D, self).__init__()
        self.teacher_discriminator_c = DiscriminatorC(ev_input_dim).double()

    def forward(self, input):
        output = self.teacher_discriminator_c(input)
        return output

class TeacherModel(nn.Module):
    def __init__(self, ev_input_dim, ev_latent_dim, es_input_dim, es_hidden_dim, dv_output_dim):
        super(TeacherModel, self).__init__()
        self.teacher_encoder_ev = EncoderEv(ev_input_dim).double()
        self.teacher_decoder_dv = DecoderDv(ev_latent_dim, dv_output_dim).double()
        self.teacher_discriminator_c = DiscriminatorC(ev_input_dim).double()
        self.selayer = SELayer(ev_latent_dim).double()

    def forward(self, f):
        z = self.teacher_encoder_ev(f)
        z_atti = self.selayer(z)
        y = self.teacher_decoder_dv(z_atti)
        return z, y

class TeacherStudentModel(nn.Module):
    def __init__(self, ev_input_dim, ev_latent_dim, es_input_dim, es_hidden_dim, dv_output_dim):
        super(TeacherStudentModel, self).__init__()
        self.teacher_encoder_ev = EncoderEv(ev_input_dim).double()
        self.teacher_decoder_dv = DecoderDv(ev_latent_dim, dv_output_dim).double()
        self.teacher_discriminator_c = DiscriminatorC(ev_input_dim).double()

        self.student_encoder_es = EncoderEs(es_input_dim, es_hidden_dim, ev_latent_dim).double()
        self.student_decoder_ds = self.teacher_decoder_dv
        self.selayer = SELayer(ev_latent_dim).double()

    def forward(self, f, a):
        z = self.teacher_encoder_ev(f)
        z_atti = self.selayer(z)
        y = self.teacher_decoder_dv(z_atti)

        v = self.student_encoder_es(a)
        v_atti = self.selayer(v)
        s = self.teacher_decoder_dv(v_atti)

        return z, y, v, s

class StudentModel(nn.Module):
    def __init__(self, dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim):
        super(StudentModel, self).__init__()
        self.student_encoder_es = EncoderEs(es_input_dim, es_hidden_dim, ev_latent_dim).double()
        self.student_decoder_ds = DecoderDv(ev_latent_dim, dv_output_dim).double()
        self.selayer = SELayer(ev_latent_dim).double()

    def forward(self, x):
        v = self.student_encoder_es(x)
        v_atti = self.selayer(v)
        s = self.student_decoder_ds(v_atti)
        return s

class identifyModel(nn.Module):
    def __init__(self,dv_output_dim,id_output_dim):
        super(identifyModel, self).__init__()
        self.l1=nn.Linear(dv_output_dim, id_output_dim, dtype=torch.double)
        self.softmax=nn.Softmax().double()
    def forward(self,r):
        tt=self.l1(r)
        res=self.softmax(tt)
        return res

