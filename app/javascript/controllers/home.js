// FILE: app/javascript/controllers/home.js
// PURPOSE: Founder modal — open/close logic + founder data.

import {founders} from "./founder_modal"

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