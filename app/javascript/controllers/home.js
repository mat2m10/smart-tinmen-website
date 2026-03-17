// FILE: app/javascript/controllers/home.js
// PURPOSE: Founder modal — open/close logic + founder data.

const founders = {
  harold: {
    name: 'Harold Loop',
    role: 'CEO',
    bio: 'Harold is an industrial electromechanical engineer with hands-on experience delivering end-to-end digitalisation projects in demanding real-world environments — power plants, robotics centres, and industrial operations. He combines deep technical grounding with sharp project delivery skills, having led multidisciplinary teams as product owner, business analyst, and on-site implementation lead across the energy and robotics sectors.',
    skills: ['Agile project management', 'CAD (SolidWorks, Catia, AutoCAD)', 'ETL design', 'SQL / Python / MATLAB', 'Stakeholder management', 'Change management', 'Prince2 · Agile Practitioner'],
    exp: [
      { title: 'Consultant — Energy Sector, Avertim', sub: 'End-to-end digitalisation in power plants · Product owner & BA · 2022–present' },
      { title: 'Data Engineer — Deloitte Consulting',  sub: 'ETL pipelines & BI · Industrial waste sector · 2021–2022' },
      { title: 'Internship & Thesis — VLIZ (Marine Robotics)', sub: 'Underwater data logger prototype · Deployed in Greenland & North Sea · 2020' },
      { title: 'MA Electromechanical Engineering — ECAM', sub: 'Magna Cum Laude · Brussels · 2020' },
      { title: 'Advanced MA Innovation & Strategic Mgmt. — Solvay ULB', sub: 'Magna Cum Laude · Brussels · 2021' },
    ]
  },
  matthieu: {
    name: 'Matthieu de Hemptinne',
    role: 'CTO',
    bio: 'Matthieu is an applied software engineer with 6+ years across robotics (ROS/ROS2), computer vision, deep learning, and production data systems. He has deployed perception and ML pipelines in constrained real-world settings spanning underwater robotics, wildlife monitoring, population-scale genetics, and healthcare analytics.',
    skills: ['ROS2 / Robotics', 'Computer vision (YOLO, Faster R-CNN)', 'PyTorch / TensorFlow', 'Python / C++', 'ETL pipelines', 'Embedded systems', 'Blender simulation'],
    skillsGreen: ['Wildlife detection', 'Ecological field systems'],
    exp: [
      { title: 'Software Engineer — Lilongwe Wildlife Centre', sub: 'Pangolin detection pipelines · Malawi · 2025–present' },
      { title: 'Software Engineer — Nkhoma Hospital', sub: 'Healthcare analytics · Malawi · 2025–present' },
      { title: 'Software Engineer — CTG (Human Genetics)', sub: 'Deep learning · GWAS · Nature Genetics · 2021–2026' },
      { title: 'Lead Teacher — Le Wagon', sub: 'ML & full-stack · 200+ students mentored · 2021–present' },
      { title: 'R&D Engineer — VLIZ', sub: 'Underwater robotics & perception · Belgium · 2019–2020' },
    ]
  },
  geoffroy: {
    name: 'Geoffroy de Cannière',
    role: 'Co-founder & Head of Forest & Field Operations',
    bio: 'Geoffroy brings deep expertise in forestry operations, connected field tools, and ecological deployment. Co-founder of Timbtrack, a GIS-based silvicultural data platform used across Europe—building connected measurement devices and forest management software for digitising field data collection.',
    skills: ['Forest management', 'GIS & silvicultural data', 'Connected field devices', 'Business development', 'Ecological strategy'],
    skillsGreen: ['Conservation technology', 'Field operations'],
    exp: [
      { title: 'Head of Development — Forestry Europe', sub: 'Timbtrack · 2022–present' },
      { title: 'Co-founder & Head of Sales — Timbtrack (IBR SA)', sub: 'Connected forestry tools · Brussels · 2017–2022' },
      { title: 'Conseiller stratégique — KICK Belgium ASBL', sub: '2019–present' },
      { title: 'Business Developer — Forestry Club de France', sub: 'Europe, US & South America · 2016–2019' },
    ]
  },
  camilla: {
    name: 'Camilla Poli',
    role: 'Co-founder & Head of Operations',
    bio: 'Camilla specialises in scaling teams and operations without losing quality or speed. She spent 3 years at Adyen designing learning and delivery systems for 1,000+ employees globally, and co-founded Thrivx to help fast-growing tech companies strengthen the human systems behind their product. At Smart Tinmen, she ensures projects run smoothly, clients are well-served, and the team operates at its best as we grow.',
    skills: ['Operations & delivery', 'Team scaling', 'Client management', 'Learning design', 'Stakeholder alignment'],
    exp: [
      { title: 'Co-founder — Thrivx', sub: 'Scaling teams in fast-growing tech · Amsterdam · 2025–present' },
      { title: 'Program Manager — Ops Academy, Adyen', sub: '1,000+ employees onboarded · 80+ workshops · 2024–2025' },
      { title: 'Program Specialist — Ops Academy, Adyen', sub: 'Global training & learning programs · 2022–2024' },
      { title: 'Learning Strategist — Lepaya', sub: 'Training optimisation · Amsterdam · 2022' },
    ]
  }
};

function openFounder(id, imgSrc) {
  const f = founders[id];
  if (!f) return;

  document.getElementById('modal-img').src = imgSrc;
  document.getElementById('modal-img').alt  = f.name;
  document.getElementById('modal-name').textContent = f.name;
  document.getElementById('modal-role').textContent = f.role;
  document.getElementById('modal-bio').textContent  = f.bio;

  const skillsEl = document.getElementById('modal-skills');
  skillsEl.innerHTML = '';
  (f.skills || []).forEach(s => {
    const span = document.createElement('span');
    span.className = 'st-modal-skill';
    span.textContent = s;
    skillsEl.appendChild(span);
  });
  (f.skillsGreen || []).forEach(s => {
    const span = document.createElement('span');
    span.className = 'st-modal-skill st-modal-skill-green';
    span.textContent = s;
    skillsEl.appendChild(span);
  });

  const expEl = document.getElementById('modal-exp');
  expEl.innerHTML = '';
  (f.exp || []).forEach(e => {
    const item = document.createElement('div');
    item.className = 'st-modal-exp-item';
    item.innerHTML = `<div class="st-modal-exp-dot"></div><div><div class="st-modal-exp-title">${e.title}</div><div class="st-modal-exp-sub">${e.sub}</div></div>`;
    expEl.appendChild(item);
  });

  document.getElementById('founderModal').classList.add('open');
}

function closeModal() {
  document.getElementById('founderModal').classList.remove('open');
}

// Event delegation on body — no DOMContentLoaded needed, works immediately
document.body.addEventListener('click', e => {
  const card = e.target.closest('.st-founder-card');
  if (card) {
    openFounder(card.dataset.founder, card.dataset.img);
    return;
  }
  if (e.target.closest('.st-modal-close')) {
    closeModal();
    return;
  }
  if (e.target.id === 'founderModal') {
    closeModal();
  }
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});