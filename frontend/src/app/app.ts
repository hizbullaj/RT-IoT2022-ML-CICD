import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { FormsModule } from '@angular/forms';

interface Feature {
  key: string;
  label: string;
  description: string;
}

interface PredictionResult {
  prediction: string;
  model: string;
  task: string;
  disclaimer: string;
  probabilities?: Record<string, number>;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {

  apiOnline = false;
  loading = false;
  errorMessage = '';
  successMessage = '';

  result: PredictionResult | null = null;

  serviceOptions = [
    '-',
    'dhcp',
    'dns',
    'http',
    'irc',
    'mqtt',
    'ntp',
    'radius',
    'ssh',
    'ssl'
  ];

  features: Feature[] = [
    {
      key: 'id.resp_p',
      label: 'Destination Port',
      description: 'Destination port number'
    },
    {
      key: 'active.min',
      label: 'Active Minimum',
      description: 'Minimum active connection time'
    },
    {
      key: 'bwd_pkts_payload.std',
      label: 'Backward Payload Std',
      description: 'Standard deviation of backward packet payload'
    },
    {
      key: 'service',
      label: 'Network Service',
      description: 'Select the detected network service'
    },
    {
      key: 'fwd_last_window_size',
      label: 'Forward Last Window Size',
      description: 'Last forward TCP window size'
    },
    {
      key: 'fwd_URG_flag_count',
      label: 'Forward URG Flag Count',
      description: 'Number of forward urgent flags'
    },
    {
      key: 'fwd_header_size_tot',
      label: 'Forward Header Size Total',
      description: 'Total size of forward headers'
    },
    {
      key: 'flow_iat.min',
      label: 'Flow IAT Minimum',
      description: 'Minimum time between flow packets'
    },
    {
      key: 'fwd_header_size_min',
      label: 'Forward Header Size Minimum',
      description: 'Minimum forward header size'
    },
    {
      key: 'bwd_pkts_payload.avg',
      label: 'Backward Payload Average',
      description: 'Average backward packet payload'
    },
    {
      key: 'fwd_iat.min',
      label: 'Forward IAT Minimum',
      description: 'Minimum forward packet interval'
    },
    {
      key: 'fwd_pkts_payload.min',
      label: 'Forward Payload Minimum',
      description: 'Minimum forward packet payload'
    },
    {
      key: 'fwd_init_window_size',
      label: 'Forward Initial Window',
      description: 'Initial forward TCP window size'
    },
    {
      key: 'flow_duration',
      label: 'Flow Duration',
      description: 'Total duration of the network flow'
    },
    {
      key: 'fwd_header_size_max',
      label: 'Forward Header Size Maximum',
      description: 'Maximum forward header size'
    }
  ];

  formData: Record<string, number | string> = {};

constructor(private cdr: ChangeDetectorRef) {
  this.resetForm();
}

  ngOnInit(): void {
    this.checkApi();
  }

  resetForm(): void {
    this.formData = {};

    for (const feature of this.features) {
      this.formData[feature.key] =
        feature.key === 'service' ? '-' : 0;
    }

    this.result = null;
    this.errorMessage = '';
    this.successMessage = '';
  }

async checkApi(): Promise<void> {
  try {
    const response = await fetch('/health');
    this.apiOnline = response.ok;
  } catch {
    this.apiOnline = false;
  } finally {
    this.cdr.detectChanges();
  }
}

  async loadSample(): Promise<void> {
    this.errorMessage = '';
    this.successMessage = '';

    try {
      const response = await fetch('/sample-input.json');

      if (!response.ok) {
        throw new Error('Could not load the sample input file.');
      }

      const sample = await response.json();

      for (const feature of this.features) {
        if (!(feature.key in sample)) {
          throw new Error(
            `Sample data is missing ${feature.key}`
          );
        }

        this.formData[feature.key] = sample[feature.key];
      }

      this.result = null;
      this.successMessage = 'Real test sample loaded successfully.';
    } catch (error) {
      this.errorMessage =
        error instanceof Error
          ? error.message
          : 'Unable to load sample data.';
    }
    this.cdr.detectChanges();
  }

  async predict(): Promise<void> {
    this.errorMessage = '';
    this.successMessage = '';
    this.result = null;

    const payload: Record<string, number | string> = {};

    for (const feature of this.features) {
      const value = this.formData[feature.key];

      if (value === '' || value === null || value === undefined) {
        this.errorMessage = `${feature.label} is required.`;
        return;
      }

      if (feature.key === 'service') {
        if (!this.serviceOptions.includes(String(value))) {
          this.errorMessage = 'Please select a valid network service.';
          return;
        }

        payload[feature.key] = String(value);
      } else {
        const numberValue = Number(value);

        if (!Number.isFinite(numberValue)) {
          this.errorMessage = `${feature.label} must be a valid number.`;
          return;
        }

        payload[feature.key] = numberValue;
      }
    }

    this.loading = true;

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction request failed.');
      }

      this.result = data;
      this.successMessage = 'Prediction completed successfully.';
    } catch (error) {
      this.errorMessage =
        error instanceof Error
          ? error.message
          : 'Could not connect to the prediction API.';

      this.apiOnline = false;
    } finally {
  this.loading = false;
  this.cdr.detectChanges();
  this.checkApi();
}
  }

  get probabilityRows(): { name: string; value: number }[] {
    if (!this.result?.probabilities) {
      return [];
    }

    return Object.entries(this.result.probabilities)
      .map(([name, value]) => ({ name, value: Number(value) }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 6);
  }

  get predictionIsAttack(): boolean {
    if (!this.result) {
      return false;
    }

    return !this.result.prediction.toLowerCase().includes('benign');
  }

  formatProbability(value: number): string {
    return (value * 100).toFixed(2) + '%';
  }
}